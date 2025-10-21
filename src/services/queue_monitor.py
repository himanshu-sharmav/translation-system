"""
Queue monitoring and management service for tracking queue performance and health.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from src.config.config import Priority, JobStatus
from src.database.connection import get_db_session
from src.database.models import QueueMetric, SystemMetric
from src.database.repositories import JobRepository, MetricsRepository
from src.database.repositories.base import BaseRepository
from src.services.queue_service import QueueService
from src.services.notification_service import NotificationService
from src.utils.logging import TranslationLogger

logger = TranslationLogger(__name__, "queue-monitor")


class QueueMonitor:
    """Service for monitoring queue performance and health."""
    
    def __init__(self):
        self.queue_service = QueueService()
        self.notification_service = NotificationService()
        self.monitoring_interval = 60  # seconds
        self.alert_thresholds = {
            "max_queue_depth": 100,
            "max_wait_time_minutes": 30,
            "min_processing_rate": 10,  # jobs per minute
            "max_dlq_size": 50
        }
        self._monitoring_task = None
        self._is_monitoring = False
    
    async def start_monitoring(self):
        """Start continuous queue monitoring."""
        if self._is_monitoring:
            logger.warning("Queue monitoring is already running")
            return
        
        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Queue monitoring started")
    
    async def stop_monitoring(self):
        """Stop queue monitoring."""
        if not self._is_monitoring:
            return
        
        self._is_monitoring = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Queue monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._is_monitoring:
            try:
                await self._collect_queue_metrics()
                await self._check_queue_health()
                await self._cleanup_stale_jobs()
                
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in queue monitoring loop: {str(e)}", exc_info=True)
                await asyncio.sleep(self.monitoring_interval)
    
    async def _collect_queue_metrics(self):
        """Collect and store queue performance metrics."""
        try:
            # Get queue statistics
            queue_stats = await self.queue_service.get_queue_stats()
            
            # Calculate processing rates and wait times
            async with get_db_session() as session:
                job_repo = JobRepository(session)
                queue_metric_repo = BaseRepository(session, QueueMetric)
                
                current_time = datetime.utcnow()
                
                # Get recent job completion data for processing rate calculation
                recent_jobs = await job_repo.find_by({
                    "status": JobStatus.COMPLETED.value,
                    "completed_at": {"gte": current_time - timedelta(minutes=5)}
                })
                
                processing_rate = len(recent_jobs) * 12  # jobs per hour (5-minute sample * 12)
                
                # Calculate average wait time for recently completed jobs
                total_wait_time = 0
                wait_time_count = 0
                
                for job in recent_jobs:
                    if job.started_at and job.created_at:
                        wait_time = (job.started_at - job.created_at).total_seconds() / 60
                        total_wait_time += wait_time
                        wait_time_count += 1
                
                avg_wait_time = total_wait_time / wait_time_count if wait_time_count > 0 else 0
                
                # Store queue metrics for each priority
                for priority in ["critical", "high", "normal"]:
                    queue_depth = queue_stats.get(priority, 0)
                    
                    queue_metric = QueueMetric(
                        timestamp=current_time,
                        priority=priority,
                        queue_depth=queue_depth,
                        processing_rate=processing_rate if priority == "normal" else 0,
                        average_wait_time=avg_wait_time if priority == "normal" else 0,
                        dead_letter_count=queue_stats.get("dead_letter", 0) if priority == "normal" else 0
                    )
                    
                    await queue_metric_repo.create(queue_metric)
                
                await session.commit()
                
                logger.debug(f"Collected queue metrics: {queue_stats}")
                
        except Exception as e:
            logger.error(f"Failed to collect queue metrics: {str(e)}", exc_info=True)
    
    async def _check_queue_health(self):
        """Check queue health and send alerts if thresholds are exceeded."""
        try:
            queue_stats = await self.queue_service.get_queue_stats()
            alerts = []
            
            # Check queue depth thresholds
            total_queued = sum(queue_stats.get(p, 0) for p in ["critical", "high", "normal"])
            if total_queued > self.alert_thresholds["max_queue_depth"]:
                alerts.append({
                    "type": "high_queue_depth",
                    "message": f"Total queue depth ({total_queued}) exceeds threshold ({self.alert_thresholds['max_queue_depth']})",
                    "severity": "warning",
                    "data": {"total_queued": total_queued, "threshold": self.alert_thresholds["max_queue_depth"]}
                })
            
            # Check dead letter queue size
            dlq_size = queue_stats.get("dead_letter", 0)
            if dlq_size > self.alert_thresholds["max_dlq_size"]:
                alerts.append({
                    "type": "high_dlq_size",
                    "message": f"Dead letter queue size ({dlq_size}) exceeds threshold ({self.alert_thresholds['max_dlq_size']})",
                    "severity": "critical",
                    "data": {"dlq_size": dlq_size, "threshold": self.alert_thresholds["max_dlq_size"]}
                })
            
            # Check processing rate
            async with get_db_session() as session:
                job_repo = JobRepository(session)
                
                # Get jobs completed in the last 10 minutes
                recent_completed = await job_repo.find_by({
                    "status": JobStatus.COMPLETED.value,
                    "completed_at": {"gte": datetime.utcnow() - timedelta(minutes=10)}
                })
                
                processing_rate = len(recent_completed) * 6  # jobs per hour
                
                if processing_rate < self.alert_thresholds["min_processing_rate"]:
                    alerts.append({
                        "type": "low_processing_rate",
                        "message": f"Processing rate ({processing_rate} jobs/hour) below threshold ({self.alert_thresholds['min_processing_rate']})",
                        "severity": "warning",
                        "data": {"processing_rate": processing_rate, "threshold": self.alert_thresholds["min_processing_rate"]}
                    })
            
            # Send alerts
            for alert in alerts:
                await self._send_alert(alert)
                
        except Exception as e:
            logger.error(f"Failed to check queue health: {str(e)}", exc_info=True)
    
    async def _send_alert(self, alert: Dict[str, Any]):
        """Send queue health alert."""
        try:
            await self.notification_service.send_system_alert(
                alert_type=alert["type"],
                message=alert["message"],
                severity=alert["severity"],
                metadata=alert.get("data", {})
            )
            
            logger.warning(f"Queue alert sent: {alert['message']}")
            
        except Exception as e:
            logger.error(f"Failed to send queue alert: {str(e)}", exc_info=True)
    
    async def _cleanup_stale_jobs(self):
        """Clean up stale processing jobs."""
        try:
            cleaned_count = await self.queue_service.cleanup_stale_processing_jobs(
                timeout_minutes=30
            )
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} stale processing jobs")
                
        except Exception as e:
            logger.error(f"Failed to cleanup stale jobs: {str(e)}", exc_info=True)
    
    async def get_queue_health_report(self) -> Dict[str, Any]:
        """Get comprehensive queue health report."""
        try:
            queue_stats = await self.queue_service.get_queue_stats()
            
            async with get_db_session() as session:
                job_repo = JobRepository(session)
                metrics_repo = MetricsRepository(session)
                
                current_time = datetime.utcnow()
                
                # Get recent metrics
                recent_metrics = await metrics_repo.get_queue_metrics(
                    start_time=current_time - timedelta(hours=1),
                    end_time=current_time
                )
                
                # Calculate trends
                avg_queue_depth = sum(m.queue_depth for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0
                avg_wait_time = sum(m.average_wait_time for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0
                avg_processing_rate = sum(m.processing_rate for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0
                
                # Get job status distribution
                status_counts = await job_repo.get_status_distribution()
                
                # Get recent error rate
                recent_failed = await job_repo.find_by({
                    "status": JobStatus.FAILED.value,
                    "updated_at": {"gte": current_time - timedelta(hours=1)}
                })
                
                recent_completed = await job_repo.find_by({
                    "status": JobStatus.COMPLETED.value,
                    "completed_at": {"gte": current_time - timedelta(hours=1)}
                })
                
                total_recent = len(recent_failed) + len(recent_completed)
                error_rate = (len(recent_failed) / total_recent * 100) if total_recent > 0 else 0
                
                return {
                    "timestamp": current_time.isoformat(),
                    "queue_stats": queue_stats,
                    "trends": {
                        "avg_queue_depth": avg_queue_depth,
                        "avg_wait_time_minutes": avg_wait_time,
                        "avg_processing_rate": avg_processing_rate,
                        "error_rate_percent": error_rate
                    },
                    "status_distribution": status_counts,
                    "health_status": self._calculate_health_status(queue_stats, avg_wait_time, error_rate),
                    "alerts": await self._get_active_alerts()
                }
                
        except Exception as e:
            logger.error(f"Failed to generate queue health report: {str(e)}", exc_info=True)
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "health_status": "unknown"
            }
    
    def _calculate_health_status(self, queue_stats: Dict, avg_wait_time: float, error_rate: float) -> str:
        """Calculate overall queue health status."""
        total_queued = sum(queue_stats.get(p, 0) for p in ["critical", "high", "normal"])
        dlq_size = queue_stats.get("dead_letter", 0)
        
        # Critical conditions
        if (dlq_size > self.alert_thresholds["max_dlq_size"] or 
            error_rate > 10 or 
            total_queued > self.alert_thresholds["max_queue_depth"] * 1.5):
            return "critical"
        
        # Warning conditions
        if (total_queued > self.alert_thresholds["max_queue_depth"] or
            avg_wait_time > self.alert_thresholds["max_wait_time_minutes"] or
            error_rate > 5 or
            dlq_size > self.alert_thresholds["max_dlq_size"] * 0.5):
            return "warning"
        
        return "healthy"
    
    async def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of active alerts."""
        # This would typically query a persistent alert store
        # For now, return empty list as alerts are sent immediately
        return []
    
    async def requeue_dead_letter_jobs(self, max_jobs: int = 10) -> int:
        """Manually requeue jobs from dead letter queue."""
        try:
            requeued_count = await self.queue_service.retry_dead_letter_jobs(
                max_retries=3,
                max_jobs=max_jobs
            )
            
            if requeued_count > 0:
                logger.info(f"Manually requeued {requeued_count} jobs from dead letter queue")
            
            return requeued_count
            
        except Exception as e:
            logger.error(f"Failed to requeue dead letter jobs: {str(e)}", exc_info=True)
            return 0
    
    async def pause_queue_processing(self, priority: Optional[Priority] = None):
        """Pause queue processing for maintenance."""
        # This would set a flag that workers check before processing
        # Implementation depends on worker architecture
        logger.info(f"Queue processing pause requested for priority: {priority or 'all'}")
    
    async def resume_queue_processing(self, priority: Optional[Priority] = None):
        """Resume queue processing after maintenance."""
        # This would clear the pause flag
        logger.info(f"Queue processing resume requested for priority: {priority or 'all'}")
    
    async def close(self):
        """Clean up resources."""
        await self.stop_monitoring()
        await self.queue_service.close()