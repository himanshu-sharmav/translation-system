"""
Notification service for sending webhooks and other notifications.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID

import httpx

from src.api.models import WebhookNotificationModel
from src.config.config import config
from src.database.connection import get_db_session
from src.database.models import TranslationJob, AuditLog
from src.database.repositories.base import BaseRepository
from src.utils.exceptions import NotificationError
from src.utils.logging import TranslationLogger

logger = TranslationLogger(__name__, "notification-service")


class NotificationService:
    """Service for handling notifications and webhooks."""
    
    def __init__(self):
        self.http_client = None
        self.max_retries = 3
        self.retry_delays = [1, 5, 15]  # seconds
        self.timeout = 30  # seconds
    
    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get HTTP client for webhook requests."""
        if not self.http_client:
            self.http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
            )
        return self.http_client
    
    async def send_job_completion_webhook(self, job: TranslationJob) -> bool:
        """Send webhook notification for job completion."""
        if not job.callback_url:
            return True  # No webhook to send
        
        try:
            # Create webhook payload
            notification = WebhookNotificationModel(
                job_id=job.id,
                status=job.status,
                completed_at=job.completed_at or datetime.utcnow(),
                result_url=f"/api/v1/jobs/{job.id}/result" if job.status == "completed" else None,
                error_message=job.error_message,
                processing_time_ms=job.processing_time_ms
            )
            
            # Send webhook with retries
            success = await self._send_webhook_with_retry(
                job.callback_url,
                notification.dict(),
                job.id
            )
            
            # Log webhook attempt
            await self._log_webhook_attempt(
                job.id,
                job.user_id,
                job.callback_url,
                notification.dict(),
                success
            )
            
            return success
            
        except Exception as e:
            logger.error(
                f"Error sending job completion webhook",
                job_id=job.id,
                metadata={
                    "callback_url": job.callback_url,
                    "error": str(e)
                },
                exc_info=True
            )
            return False
    
    async def _send_webhook_with_retry(self, url: str, payload: Dict[str, Any], job_id: UUID) -> bool:
        """Send webhook with retry logic."""
        http_client = await self._get_http_client()
        
        for attempt in range(self.max_retries):
            try:
                response = await http_client.post(
                    url,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": "Translation-API-Webhook/1.0",
                        "X-Job-ID": str(job_id)
                    }
                )
                
                if response.status_code in [200, 201, 202, 204]:
                    logger.info(
                        f"Webhook sent successfully",
                        job_id=job_id,
                        metadata={
                            "url": url,
                            "status_code": response.status_code,
                            "attempt": attempt + 1
                        }
                    )
                    return True
                else:
                    logger.warning(
                        f"Webhook failed with status {response.status_code}",
                        job_id=job_id,
                        metadata={
                            "url": url,
                            "status_code": response.status_code,
                            "response_text": response.text[:500],
                            "attempt": attempt + 1
                        }
                    )
                    
            except httpx.TimeoutException:
                logger.warning(
                    f"Webhook timeout",
                    job_id=job_id,
                    metadata={
                        "url": url,
                        "attempt": attempt + 1,
                        "timeout": self.timeout
                    }
                )
                
            except httpx.RequestError as e:
                logger.warning(
                    f"Webhook request error: {str(e)}",
                    job_id=job_id,
                    metadata={
                        "url": url,
                        "attempt": attempt + 1,
                        "error": str(e)
                    }
                )
            
            except Exception as e:
                logger.error(
                    f"Unexpected webhook error: {str(e)}",
                    job_id=job_id,
                    metadata={
                        "url": url,
                        "attempt": attempt + 1
                    },
                    exc_info=True
                )
            
            # Wait before retry (except on last attempt)
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delays[attempt])
        
        logger.error(
            f"Webhook failed after {self.max_retries} attempts",
            job_id=job_id,
            metadata={"url": url}
        )
        return False
    
    async def _log_webhook_attempt(self, job_id: UUID, user_id: str, url: str, 
                                 payload: Dict[str, Any], success: bool):
        """Log webhook attempt to audit log."""
        try:
            async with get_db_session() as session:
                audit_repo = BaseRepository(session, AuditLog)
                
                audit_log = AuditLog(
                    user_id=user_id,
                    action="webhook_notification",
                    resource_type="translation_job",
                    resource_id=str(job_id),
                    request_data={
                        "webhook_url": url,
                        "payload": payload,
                        "success": success
                    },
                    response_status=200 if success else 500
                )
                
                await audit_repo.create(audit_log)
                await session.commit()
                
        except Exception as e:
            logger.error(f"Error logging webhook attempt: {str(e)}", exc_info=True)
    
    async def send_batch_completion_notification(self, batch_id: UUID, job_ids: List[UUID], 
                                               user_id: str) -> bool:
        """Send notification for batch completion."""
        try:
            # Get job statuses
            completed_jobs = 0
            failed_jobs = 0
            total_processing_time = 0
            
            async with get_db_session() as session:
                job_repo = BaseRepository(session, TranslationJob)
                
                for job_id in job_ids:
                    job = await job_repo.get_by_id(job_id)
                    if job:
                        if job.status == "completed":
                            completed_jobs += 1
                        elif job.status == "failed":
                            failed_jobs += 1
                        
                        if job.processing_time_ms:
                            total_processing_time += job.processing_time_ms
            
            # Create batch completion notification
            notification_data = {
                "batch_id": str(batch_id),
                "total_jobs": len(job_ids),
                "completed_jobs": completed_jobs,
                "failed_jobs": failed_jobs,
                "success_rate": completed_jobs / len(job_ids) if job_ids else 0,
                "total_processing_time_ms": total_processing_time,
                "completed_at": datetime.utcnow().isoformat()
            }
            
            logger.info(
                f"Batch processing completed",
                metadata={
                    "batch_id": str(batch_id),
                    "user_id": user_id,
                    **notification_data
                }
            )
            
            # Here you could send email, push notification, etc.
            # For now, we'll just log it
            
            return True
            
        except Exception as e:
            logger.error(
                f"Error sending batch completion notification",
                metadata={
                    "batch_id": str(batch_id),
                    "user_id": user_id,
                    "error": str(e)
                },
                exc_info=True
            )
            return False
    
    async def send_system_alert(self, alert_type: str, message: str, 
                              severity: str = "warning", metadata: Dict[str, Any] = None) -> bool:
        """Send system alert notification."""
        try:
            alert_data = {
                "alert_type": alert_type,
                "message": message,
                "severity": severity,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": metadata or {}
            }
            
            # Log the alert
            logger.warning(
                f"System alert: {alert_type}",
                event="system_alert",
                metadata=alert_data
            )
            
            # Here you would integrate with alerting systems like:
            # - PagerDuty
            # - Slack
            # - Email
            # - SMS
            
            # For now, we'll simulate sending to a webhook
            if hasattr(config.monitoring, 'alert_webhook_url') and config.monitoring.alert_webhook_url:
                return await self._send_webhook_with_retry(
                    config.monitoring.alert_webhook_url,
                    alert_data,
                    UUID('00000000-0000-0000-0000-000000000000')  # System alert UUID
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending system alert: {str(e)}", exc_info=True)
            return False
    
    async def send_rate_limit_notification(self, user_id: str, endpoint: str, 
                                         current_count: int, limit: int) -> bool:
        """Send rate limit exceeded notification."""
        try:
            notification_data = {
                "user_id": user_id,
                "endpoint": endpoint,
                "current_count": current_count,
                "limit": limit,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.warning(
                f"Rate limit exceeded",
                event="rate_limit_exceeded",
                metadata=notification_data
            )
            
            # Could send email to user or admin notification
            return True
            
        except Exception as e:
            logger.error(f"Error sending rate limit notification: {str(e)}", exc_info=True)
            return False
    
    async def send_cost_alert(self, daily_cost: float, threshold: float, 
                            breakdown: Dict[str, float]) -> bool:
        """Send cost threshold alert."""
        try:
            alert_data = {
                "daily_cost_usd": daily_cost,
                "threshold_usd": threshold,
                "overage_usd": daily_cost - threshold,
                "cost_breakdown": breakdown,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return await self.send_system_alert(
                "cost_threshold_exceeded",
                f"Daily cost ${daily_cost:.2f} exceeded threshold ${threshold:.2f}",
                "critical",
                alert_data
            )
            
        except Exception as e:
            logger.error(f"Error sending cost alert: {str(e)}", exc_info=True)
            return False
    
    async def send_performance_alert(self, metric_name: str, current_value: float, 
                                   threshold: float, instance_id: str = None) -> bool:
        """Send performance metric alert."""
        try:
            alert_data = {
                "metric_name": metric_name,
                "current_value": current_value,
                "threshold": threshold,
                "instance_id": instance_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            severity = "critical" if current_value > threshold * 1.5 else "warning"
            
            return await self.send_system_alert(
                "performance_threshold_exceeded",
                f"{metric_name} value {current_value} exceeded threshold {threshold}",
                severity,
                alert_data
            )
            
        except Exception as e:
            logger.error(f"Error sending performance alert: {str(e)}", exc_info=True)
            return False
    
    async def validate_webhook_url(self, url: str) -> bool:
        """Validate webhook URL by sending a test request."""
        try:
            http_client = await self._get_http_client()
            
            test_payload = {
                "test": True,
                "message": "Webhook validation test",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            response = await http_client.post(
                url,
                json=test_payload,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "Translation-API-Webhook/1.0",
                    "X-Test-Webhook": "true"
                }
            )
            
            return response.status_code in [200, 201, 202, 204]
            
        except Exception as e:
            logger.warning(f"Webhook validation failed for {url}: {str(e)}")
            return False
    
    async def close(self):
        """Close HTTP client."""
        if self.http_client:
            await self.http_client.aclose()