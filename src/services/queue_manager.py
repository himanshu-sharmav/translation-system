"""
Advanced queue management utilities and monitoring for the translation system.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from uuid import UUID

import redis.asyncio as redis

from src.config.config import config, Priority, JobStatus
from src.database.connection import get_db_session
from src.database.models import TranslationJob, QueueMetric
from src.database.repositories import JobRepository, MetricsRepository
from src.database.repositories.base import BaseRepository
from src.services.queue_service import QueueService
from src.utils.exceptions import QueueError
from src.utils.logging import TranslationLogger

logger = TranslationLogger(__name__, "queue-manager")


class QueueManager:
    """Advanced queue management with monitoring and optimization."""
    
    def __init__(self):
        self.queue_service = QueueService()
        self.redis_client = None
        self.metrics_collection_interval = 60  # seconds
        self.queue_optimization_interval = 300  # 5 minutes
        self._monitoring_task = None
        self._optimization_task = None
    
    async def _get_redis_client(self) -> redis.Redis:
        """Get Redis client connection."""
        if not self.redis_client:
            self.redis_client = redis.Redis(
                host=config.redis.host,
                port=config.redis.port,
                password=config.redis.password,
                db=config.redis.db,
                max_connections=config.redis.max_connections,
                decode_responses=True
            )
        return self.redis_client
    
    async def start_monitoring(self):
        """Start queue monitoring and optimization tasks."""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitor_queues())
            logger.info("Queue monitoring started")
        
        if self._optimization_task is None or self._optimization_task.done():
            self._optimization_task = asyncio.create_task(self._optimize_queues())
            logger.info("Queue optimization started")
    
    async def stop_monitoring(self):
        """Stop queue monitoring and optimization tasks."""
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            logger.info("Queue monitoring stopped")
        
        if self._optimization_task and not self._optimization_task.done():
            self._optimization_task.cancel()
            try:
                await self._optimization_task
            except asyncio.CancelledError:
                pass
            logger.info("Queue optimization stopped")
    
    async def _monitor_queues(self):
        """Continuously monitor queue metrics."""
        while True:
            try:
                await self._collect_queue_metrics()
                await asyncio.sleep(self.metrics_collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in queue monitoring: {str(e)}", exc_info=True)
                await asyncio.sleep(self.metrics_collection_interval)
    
    async def _optimize_queues(self):
        """Continuously optimize queue performance."""
        while True:
            try:
                await self._perform_queue_optimization()
                await asyncio.sleep(self.queue_optimization_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in queue optimization: {str(e)}", exc_info=True)
                await asyncio.sleep(self.queue_optimization_interval)
    
    async def _collect_queue_metrics(self):
        """Collect and store queue performance metrics."""
        try:
            redis_client = await self._get_redis_client()
            current_time = datetime.utcnow()
            
            # Get queue depths for each priority
            queue_depths = {}
            total_depth = 0
            
            for priority in Priority:
                queue_name = self.queue_service.queue_names[priority]
                depth = await redis_client.zcard(queue_name)
                queue_depths[priority.value] = depth
                total_depth += depth
            
            # Get processing count
            processing_count = await redis_client.scard(self.queue_service.processing_set)
            
            # Calculate wait times and processing rates
            async with get_db_session() as session:
                job_repo = JobRepository(session)
                metrics_repo = MetricsRepository(session)
                queue_metric_repo = BaseRepository(session, QueueMetric)
                
                # Calculate average wait times for each priority
                for priority in Priority:
                    if queue_depths[priority.value] > 0:
                        # Get jobs in this priority queue
                        queued_jobs = await job_repo.get_jobs_by_priority_queue(priority, limit=100)
                        
                        if queued_jobs:
                            # Calculate average wait time
                            wait_times = []
                            for job in queued_jobs:
                                wait_time = (current_time - job.created_at).total_seconds()
                                wait_times.append(wait_time)
                            
                            avg_wait_time = sum(wait_times) / len(wait_times)
                        else:
                            avg_wait_time = 0
                        
                        # Calculate processing rate (jobs per minute)
                        one_hour_ago = current_time - timedelta(hours=1)
                        recent_completed = await job_repo.find_by({
                            "status": JobStatus.COMPLETED.value,
                            "priority": priority.value,
                            "completed_at": {"gte": one_hour_ago}
                        })
                        
                        processing_rate = len(recent_completed) * 60  # per hour to per minute
                        
                        # Store queue metrics
                        queue_metric = QueueMetric(
                            timestamp=current_time,
                            priority=priority.value,
                            queue_depth=queue_depths[priority.value],
                            avg_wait_time_seconds=avg_wait_time,
                            processing_rate_per_minute=processing_rate
                        )
                        
                        await queue_metric_repo.create(queue_metric)
                
                # Store system-wide metrics
                system_metrics = [
                    {
                        "metric_name": "queue_depth_total",
                        "metric_value": total_depth,
                        "timestamp": current_time
                    },
                    {
                        "metric_name": "processing_jobs_count",
                        "metric_value": processing_count,
                        "timestamp": current_time
                    }
                ]
                
                await metrics_repo.record_bulk_metrics(system_metrics)
                await session.commit()
                
                logger.debug(
                    f"Queue metrics collected",
                    metrics={
                        "total_queue_depth": total_depth,
                        "processing_count": processing_count,
                        "critical_depth": queue_depths.get("critical", 0),
                        "high_depth": queue_depths.get("high", 0),
                        "normal_depth": queue_depths.get("normal", 0)
                    }
                )
                
        except Exception as e:
            logger.error(f"Error collecting queue metrics: {str(e)}", exc_info=True)
    
    async def _perform_queue_optimization(self):
        """Perform queue optimization tasks."""
        try:
            # Clean up stale processing jobs
            stale_count = await self.queue_service.cleanup_stale_processing_jobs(timeout_minutes=30)
            if stale_count > 0:
                logger.info(f"Cleaned up {stale_count} stale processing jobs")
            
            # Retry dead letter queue jobs
            retried_count = await self.queue_service.retry_dead_letter_jobs(max_retries=3)
            if retried_count > 0:
                logger.info(f"Retried {retried_count} jobs from dead letter queue")
            
            # Rebalance queues if needed
            await self._rebalance_queues()
            
            # Check for queue health issues
            await self._check_queue_health()
            
        except Exception as e:
            logger.error(f"Error in queue optimization: {str(e)}", exc_info=True)
    
    async def _rebalance_queues(self):
        """Rebalance queues based on processing capacity and priorities."""
        try:
            redis_client = await self._get_redis_client()
            
            # Get current queue depths
            queue_depths = {}
            for priority in Priority:
                queue_name = self.queue_service.queue_names[priority]
                depth = await redis_client.zcard(queue_name)
                queue_depths[priority] = depth
            
            # Check if rebalancing is needed
            total_jobs = sum(queue_depths.values())
            if total_jobs == 0:
                return
            
            # Get processing capacity metrics
            processing_count = await redis_client.scard(self.queue_service.processing_set)
            
            # Simple rebalancing logic: if critical queue is too full compared to processing capacity
            critical_depth = queue_depths[Priority.CRITICAL]
            high_depth = queue_depths[Priority.HIGH]
            
            if critical_depth > 10 and processing_count < 5:
                # Consider promoting some high priority jobs to critical
                # This is a simplified example - in practice, this would be more sophisticated
                logger.info(
                    f"Queue rebalancing considered",
                    metadata={
                        "critical_depth": critical_depth,
                        "high_depth": high_depth,
                        "processing_count": processing_count
                    }
                )
            
        except Exception as e:
            logger.error(f"Error rebalancing queues: {str(e)}", exc_info=True)
    
    async def _check_queue_health(self):
        """Check queue health and trigger alerts if needed."""
        try:
            redis_client = await self._get_redis_client()
            
            # Check for queue depth alerts
            for priority in Priority:
                queue_name = self.queue_service.queue_names[priority]
                depth = await redis_client.zcard(queue_name)
                
                # Define thresholds
                thresholds = {
                    Priority.CRITICAL: 5,
                    Priority.HIGH: 20,
                    Priority.NORMAL: 100
                }
                
                if depth > thresholds[priority]:
                    logger.warning(
                        f"Queue depth alert: {priority.value} queue has {depth} jobs",
                        event="queue_depth_alert",
                        metadata={
                            "priority": priority.value,
                            "depth": depth,
                            "threshold": thresholds[priority]
                        }
                    )
            
            # Check for stale jobs in processing
            processing_jobs = await redis_client.smembers(self.queue_service.processing_set)
            if len(processing_jobs) > 0:
                # Check if any jobs have been processing too long
                async with get_db_session() as session:
                    job_repo = JobRepository(session)
                    
                    for job_id_str in processing_jobs:
                        job_id = UUID(job_id_str)
                        job = await job_repo.get_by_id(job_id)
                        
                        if job and job.started_at:
                            processing_time = datetime.utcnow() - job.started_at
                            if processing_time > timedelta(minutes=15):  # Alert threshold
                                logger.warning(
                                    f"Long-running job detected",
                                    job_id=job_id,
                                    event="long_running_job",
                                    metadata={
                                        "processing_time_minutes": processing_time.total_seconds() / 60,
                                        "job_priority": job.priority
                                    }
                                )
            
        except Exception as e:
            logger.error(f"Error checking queue health: {str(e)}", exc_info=True)
    
    async def get_queue_analytics(self, hours: int = 24) -> Dict:
        """Get comprehensive queue analytics."""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)
            
            async with get_db_session() as session:
                queue_metric_repo = BaseRepository(session, QueueMetric)
                job_repo = JobRepository(session)
                
                # Get queue metrics for the time period
                queue_metrics = await queue_metric_repo.find_by({
                    "timestamp": {"gte": start_time, "lte": end_time}
                })
                
                # Organize metrics by priority
                analytics = {
                    "time_period_hours": hours,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "priority_analytics": {},
                    "system_analytics": {}
                }
                
                for priority in Priority:
                    priority_metrics = [m for m in queue_metrics if m.priority == priority.value]
                    
                    if priority_metrics:
                        avg_depth = sum(m.queue_depth for m in priority_metrics) / len(priority_metrics)
                        max_depth = max(m.queue_depth for m in priority_metrics)
                        avg_wait_time = sum(m.avg_wait_time_seconds or 0 for m in priority_metrics) / len(priority_metrics)
                        avg_processing_rate = sum(m.processing_rate_per_minute or 0 for m in priority_metrics) / len(priority_metrics)
                        
                        analytics["priority_analytics"][priority.value] = {
                            "avg_queue_depth": round(avg_depth, 2),
                            "max_queue_depth": max_depth,
                            "avg_wait_time_seconds": round(avg_wait_time, 2),
                            "avg_processing_rate_per_minute": round(avg_processing_rate, 2),
                            "total_metrics_points": len(priority_metrics)
                        }
                
                # Get job completion statistics
                completed_jobs = await job_repo.find_by({
                    "status": JobStatus.COMPLETED.value,
                    "completed_at": {"gte": start_time, "lte": end_time}
                })
                
                failed_jobs = await job_repo.find_by({
                    "status": JobStatus.FAILED.value,
                    "completed_at": {"gte": start_time, "lte": end_time}
                })
                
                # Calculate system analytics
                total_completed = len(completed_jobs)
                total_failed = len(failed_jobs)
                success_rate = total_completed / (total_completed + total_failed) if (total_completed + total_failed) > 0 else 0
                
                if completed_jobs:
                    avg_processing_time = sum(job.processing_time_ms or 0 for job in completed_jobs) / len(completed_jobs)
                    total_words = sum(job.word_count for job in completed_jobs)
                    avg_words_per_minute = (total_words / (avg_processing_time / 1000 / 60)) if avg_processing_time > 0 else 0
                else:
                    avg_processing_time = 0
                    total_words = 0
                    avg_words_per_minute = 0
                
                analytics["system_analytics"] = {
                    "total_completed_jobs": total_completed,
                    "total_failed_jobs": total_failed,
                    "success_rate": round(success_rate, 3),
                    "avg_processing_time_ms": round(avg_processing_time, 2),
                    "total_words_processed": total_words,
                    "avg_words_per_minute": round(avg_words_per_minute, 2),
                    "throughput_jobs_per_hour": round(total_completed / hours, 2)
                }
                
                return analytics
                
        except Exception as e:
            logger.error(f"Error getting queue analytics: {str(e)}", exc_info=True)
            raise QueueError(f"Failed to get queue analytics: {str(e)}")
    
    async def get_current_queue_status(self) -> Dict:
        """Get current real-time queue status."""
        try:
            redis_client = await self._get_redis_client()
            
            # Get current queue depths
            queue_status = {
                "timestamp": datetime.utcnow().isoformat(),
                "queues": {},
                "processing": {},
                "dead_letter_queue": {}
            }
            
            total_queued = 0
            for priority in Priority:
                queue_name = self.queue_service.queue_names[priority]
                depth = await redis_client.zcard(queue_name)
                queue_status["queues"][priority.value] = {
                    "depth": depth,
                    "queue_name": queue_name
                }
                total_queued += depth
            
            # Get processing status
            processing_count = await redis_client.scard(self.queue_service.processing_set)
            processing_jobs = await redis_client.smembers(self.queue_service.processing_set)
            
            queue_status["processing"] = {
                "count": processing_count,
                "job_ids": list(processing_jobs)
            }
            
            # Get dead letter queue status
            dlq_count = await redis_client.llen(self.queue_service.dead_letter_queue)
            queue_status["dead_letter_queue"] = {
                "count": dlq_count
            }
            
            # Add summary
            queue_status["summary"] = {
                "total_queued": total_queued,
                "total_processing": processing_count,
                "total_failed": dlq_count,
                "total_active": total_queued + processing_count
            }
            
            return queue_status
            
        except Exception as e:
            logger.error(f"Error getting queue status: {str(e)}", exc_info=True)
            raise QueueError(f"Failed to get queue status: {str(e)}")
    
    async def emergency_queue_drain(self, priority: Optional[Priority] = None) -> Dict:
        """Emergency procedure to drain queues (for maintenance)."""
        try:
            redis_client = await self._get_redis_client()
            
            drained_jobs = []
            priorities_to_drain = [priority] if priority else list(Priority)
            
            for p in priorities_to_drain:
                queue_name = self.queue_service.queue_names[p]
                
                # Get all jobs from queue
                jobs = await redis_client.zrange(queue_name, 0, -1)
                
                for job_data_str in jobs:
                    job_data = json.loads(job_data_str)
                    job_id = UUID(job_data["job_id"])
                    
                    # Update job status to failed with maintenance message
                    async with get_db_session() as session:
                        job_repo = JobRepository(session)
                        await job_repo.update_job_status(
                            job_id,
                            JobStatus.FAILED,
                            error_message="System maintenance - job cancelled"
                        )
                        await session.commit()
                    
                    drained_jobs.append(str(job_id))
                
                # Clear the queue
                await redis_client.delete(queue_name)
            
            logger.warning(
                f"Emergency queue drain completed",
                event="emergency_queue_drain",
                metadata={
                    "drained_priorities": [p.value for p in priorities_to_drain],
                    "drained_job_count": len(drained_jobs)
                }
            )
            
            return {
                "drained_priorities": [p.value for p in priorities_to_drain],
                "drained_jobs": drained_jobs,
                "drained_count": len(drained_jobs),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in emergency queue drain: {str(e)}", exc_info=True)
            raise QueueError(f"Failed to drain queues: {str(e)}")
    
    async def close(self):
        """Close queue manager and cleanup resources."""
        await self.stop_monitoring()
        await self.queue_service.close()
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Queue manager closed")