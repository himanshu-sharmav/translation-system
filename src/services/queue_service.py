"""
Queue service for managing translation job queues with priority handling.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from uuid import UUID

import redis.asyncio as redis

from src.config.config import config, Priority, JobStatus
from src.database.connection import get_db_session
from src.database.models import TranslationJob
from src.database.repositories import JobRepository
from src.models.interfaces import QueueManager
from src.utils.exceptions import QueueError
from src.utils.logging import TranslationLogger

logger = TranslationLogger(__name__, "queue-service")


class QueueService(QueueManager):
    """Redis-based queue service with priority handling."""
    
    def __init__(self):
        self.redis_client = None
        self.queue_names = {
            Priority.CRITICAL: "queue:critical",
            Priority.HIGH: "queue:high", 
            Priority.NORMAL: "queue:normal"
        }
        self.processing_set = "processing_jobs"
        self.dead_letter_queue = "queue:dlq"
    
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
    
    async def enqueue(self, job: TranslationJob) -> bool:
        """Add a job to the appropriate priority queue."""
        try:
            redis_client = await self._get_redis_client()
            
            # Determine queue based on priority
            priority = Priority(job.priority)
            queue_name = self.queue_names[priority]
            
            # Serialize job data
            job_data = {
                "job_id": str(job.id),
                "user_id": job.user_id,
                "source_language": job.source_language,
                "target_language": job.target_language,
                "content_hash": job.content_hash,
                "word_count": job.word_count,
                "priority": job.priority,
                "created_at": job.created_at.isoformat(),
                "callback_url": job.callback_url
            }
            
            # Add to queue with timestamp score for FIFO within priority
            score = job.created_at.timestamp()
            await redis_client.zadd(queue_name, {json.dumps(job_data): score})
            
            logger.info(
                f"Job enqueued successfully",
                job_id=job.id,
                metadata={
                    "priority": job.priority,
                    "queue_name": queue_name,
                    "word_count": job.word_count
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error enqueuing job {job.id}: {str(e)}", exc_info=True)
            raise QueueError(f"Failed to enqueue job: {str(e)}")
    
    async def dequeue(self, priority: Optional[Priority] = None) -> Optional[TranslationJob]:
        """Get the next job from the queue based on priority."""
        try:
            redis_client = await self._get_redis_client()
            
            # Define queue order (critical -> high -> normal)
            queue_order = [Priority.CRITICAL, Priority.HIGH, Priority.NORMAL]
            
            if priority:
                # Only check specific priority queue
                queue_order = [priority]
            
            for queue_priority in queue_order:
                queue_name = self.queue_names[queue_priority]
                
                # Get oldest job from this priority queue
                result = await redis_client.zpopmin(queue_name, count=1)
                
                if result:
                    job_data_str, score = result[0]
                    job_data = json.loads(job_data_str)
                    
                    # Convert back to TranslationJob object
                    job = TranslationJob(
                        id=UUID(job_data["job_id"]),
                        user_id=job_data["user_id"],
                        source_language=job_data["source_language"],
                        target_language=job_data["target_language"],
                        content_hash=job_data["content_hash"],
                        word_count=job_data["word_count"],
                        priority=job_data["priority"],
                        status=JobStatus.QUEUED.value,
                        created_at=datetime.fromisoformat(job_data["created_at"]),
                        callback_url=job_data.get("callback_url")
                    )
                    
                    # Add to processing set
                    await redis_client.sadd(self.processing_set, str(job.id))
                    
                    logger.info(
                        f"Job dequeued successfully",
                        job_id=job.id,
                        metadata={
                            "priority": job.priority,
                            "queue_name": queue_name
                        }
                    )
                    
                    return job
            
            # No jobs available in any queue
            return None
            
        except Exception as e:
            logger.error(f"Error dequeuing job: {str(e)}", exc_info=True)
            raise QueueError(f"Failed to dequeue job: {str(e)}")
    
    async def get_queue_depth(self, priority: Optional[Priority] = None) -> int:
        """Get the number of jobs in queue."""
        try:
            redis_client = await self._get_redis_client()
            
            if priority:
                # Get depth for specific priority
                queue_name = self.queue_names[priority]
                return await redis_client.zcard(queue_name)
            else:
                # Get total depth across all queues
                total_depth = 0
                for queue_name in self.queue_names.values():
                    depth = await redis_client.zcard(queue_name)
                    total_depth += depth
                return total_depth
                
        except Exception as e:
            logger.error(f"Error getting queue depth: {str(e)}", exc_info=True)
            raise QueueError(f"Failed to get queue depth: {str(e)}")
    
    async def update_job_status(self, job_id: UUID, status: JobStatus, progress: float = None) -> bool:
        """Update job status and progress."""
        try:
            async with get_db_session() as session:
                job_repo = JobRepository(session)
                success = await job_repo.update_job_status(
                    job_id, status, progress
                )
                await session.commit()
                
                # Remove from processing set if completed or failed
                if status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                    redis_client = await self._get_redis_client()
                    await redis_client.srem(self.processing_set, str(job_id))
                
                return success
                
        except Exception as e:
            logger.error(f"Error updating job status for {job_id}: {str(e)}", exc_info=True)
            return False
    
    async def get_job(self, job_id: UUID) -> Optional[TranslationJob]:
        """Get job details by ID."""
        try:
            async with get_db_session() as session:
                job_repo = JobRepository(session)
                return await job_repo.get_by_id(job_id)
                
        except Exception as e:
            logger.error(f"Error getting job {job_id}: {str(e)}", exc_info=True)
            return None
    
    async def remove_job(self, job_id: UUID) -> bool:
        """Remove job from queue (for cancellation)."""
        try:
            redis_client = await self._get_redis_client()
            
            # Try to remove from all priority queues
            removed = False
            for queue_name in self.queue_names.values():
                # Get all jobs in queue
                jobs = await redis_client.zrange(queue_name, 0, -1)
                
                for job_data_str in jobs:
                    job_data = json.loads(job_data_str)
                    if job_data["job_id"] == str(job_id):
                        # Remove this job
                        await redis_client.zrem(queue_name, job_data_str)
                        removed = True
                        break
                
                if removed:
                    break
            
            # Also remove from processing set
            await redis_client.srem(self.processing_set, str(job_id))
            
            if removed:
                logger.info(
                    f"Job removed from queue",
                    job_id=job_id
                )
            
            return removed
            
        except Exception as e:
            logger.error(f"Error removing job {job_id} from queue: {str(e)}", exc_info=True)
            return False
    
    async def enqueue_job(self, job: TranslationJob) -> bool:
        """Convenience method to enqueue a job."""
        return await self.enqueue(job)
    
    async def get_queue_stats(self) -> Dict[str, int]:
        """Get statistics for all queues."""
        try:
            redis_client = await self._get_redis_client()
            
            stats = {}
            for priority, queue_name in self.queue_names.items():
                depth = await redis_client.zcard(queue_name)
                stats[priority.value] = depth
            
            # Get processing count
            processing_count = await redis_client.scard(self.processing_set)
            stats["processing"] = processing_count
            
            # Get dead letter queue count
            dlq_count = await redis_client.llen(self.dead_letter_queue)
            stats["dead_letter"] = dlq_count
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting queue stats: {str(e)}", exc_info=True)
            return {}
    
    async def move_to_dead_letter_queue(self, job_id: UUID, error_message: str) -> bool:
        """Move failed job to dead letter queue."""
        try:
            redis_client = await self._get_redis_client()
            
            # Get job details
            job = await self.get_job(job_id)
            if not job:
                return False
            
            # Create DLQ entry
            dlq_entry = {
                "job_id": str(job_id),
                "error_message": error_message,
                "failed_at": datetime.utcnow().isoformat(),
                "original_priority": job.priority,
                "retry_count": 0
            }
            
            # Add to dead letter queue
            await redis_client.lpush(self.dead_letter_queue, json.dumps(dlq_entry))
            
            # Remove from processing set
            await redis_client.srem(self.processing_set, str(job_id))
            
            logger.warning(
                f"Job moved to dead letter queue",
                job_id=job_id,
                metadata={
                    "error_message": error_message,
                    "priority": job.priority
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error moving job {job_id} to DLQ: {str(e)}", exc_info=True)
            return False
    
    async def retry_dead_letter_jobs(self, max_retries: int = 3) -> int:
        """Retry jobs from dead letter queue."""
        try:
            redis_client = await self._get_redis_client()
            
            retried_count = 0
            dlq_length = await redis_client.llen(self.dead_letter_queue)
            
            for _ in range(min(dlq_length, 100)):  # Process max 100 at a time
                dlq_entry_str = await redis_client.rpop(self.dead_letter_queue)
                if not dlq_entry_str:
                    break
                
                dlq_entry = json.loads(dlq_entry_str)
                retry_count = dlq_entry.get("retry_count", 0)
                
                if retry_count < max_retries:
                    # Get job and re-enqueue
                    job_id = UUID(dlq_entry["job_id"])
                    job = await self.get_job(job_id)
                    
                    if job and job.status == JobStatus.FAILED.value:
                        # Reset job status and re-enqueue
                        await self.update_job_status(job_id, JobStatus.QUEUED)
                        await self.enqueue(job)
                        retried_count += 1
                        
                        logger.info(
                            f"Job retried from DLQ",
                            job_id=job_id,
                            metadata={
                                "retry_count": retry_count + 1,
                                "max_retries": max_retries
                            }
                        )
                    else:
                        # Put back in DLQ with incremented retry count
                        dlq_entry["retry_count"] = retry_count + 1
                        await redis_client.lpush(self.dead_letter_queue, json.dumps(dlq_entry))
                else:
                    # Max retries reached, keep in DLQ
                    await redis_client.lpush(self.dead_letter_queue, dlq_entry_str)
            
            logger.info(
                f"Dead letter queue processing completed",
                metadata={
                    "retried_count": retried_count,
                    "remaining_dlq_size": await redis_client.llen(self.dead_letter_queue)
                }
            )
            
            return retried_count
            
        except Exception as e:
            logger.error(f"Error retrying DLQ jobs: {str(e)}", exc_info=True)
            return 0
    
    async def cleanup_stale_processing_jobs(self, timeout_minutes: int = 30) -> int:
        """Clean up jobs that have been processing too long."""
        try:
            redis_client = await self._get_redis_client()
            
            # Get all processing job IDs
            processing_job_ids = await redis_client.smembers(self.processing_set)
            
            cleaned_count = 0
            for job_id_str in processing_job_ids:
                job_id = UUID(job_id_str)
                
                # Check job status in database
                async with get_db_session() as session:
                    job_repo = JobRepository(session)
                    job = await job_repo.get_by_id(job_id)
                    
                    if not job:
                        # Job not found, remove from processing set
                        await redis_client.srem(self.processing_set, job_id_str)
                        cleaned_count += 1
                        continue
                    
                    # Check if job has been processing too long
                    if (job.status == JobStatus.PROCESSING.value and 
                        job.started_at and 
                        datetime.utcnow() - job.started_at > timedelta(minutes=timeout_minutes)):
                        
                        # Move to failed status and DLQ
                        await job_repo.update_job_status(
                            job_id, 
                            JobStatus.FAILED,
                            error_message=f"Job timed out after {timeout_minutes} minutes"
                        )
                        await session.commit()
                        
                        await self.move_to_dead_letter_queue(
                            job_id, 
                            f"Processing timeout after {timeout_minutes} minutes"
                        )
                        
                        cleaned_count += 1
                        
                        logger.warning(
                            f"Stale processing job cleaned up",
                            job_id=job_id,
                            metadata={
                                "timeout_minutes": timeout_minutes,
                                "processing_time": str(datetime.utcnow() - job.started_at)
                            }
                        )
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up stale processing jobs: {str(e)}", exc_info=True)
            return 0
    
    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()