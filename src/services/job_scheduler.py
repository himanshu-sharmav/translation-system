"""
Job scheduling and dispatch service for the translation system.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from uuid import UUID

from src.config.config import config, Priority, JobStatus
from src.database.connection import get_db_session
from src.database.models import TranslationJob, ComputeInstance
from src.database.repositories import JobRepository
from src.database.repositories.base import BaseRepository
from src.services.queue_service import QueueService
from src.services.notification_service import NotificationService
from src.utils.exceptions import QueueError, JobTimeoutError
from src.utils.logging import TranslationLogger

logger = TranslationLogger(__name__, "job-scheduler")


class JobScheduler:
    """Job scheduling and dispatch service with intelligent routing."""
    
    def __init__(self):
        self.queue_service = QueueService()
        self.notification_service = NotificationService()
        self.active_dispatchers: Set[str] = set()
        self.dispatch_interval = 1.0  # seconds
        self.max_concurrent_dispatchers = 5
        self.job_timeout_minutes = 30
        self._dispatch_task = None
        self._timeout_check_task = None
        self._running = False
    
    async def start_dispatcher(self):
        """Start the job dispatcher."""
        if self._running:
            logger.warning("Job dispatcher is already running")
            return
        
        self._running = True
        self._dispatch_task = asyncio.create_task(self._dispatch_loop())
        self._timeout_check_task = asyncio.create_task(self._timeout_check_loop())
        
        logger.info("Job dispatcher started")
    
    async def stop_dispatcher(self):
        """Stop the job dispatcher."""
        if not self._running:
            return
        
        self._running = False
        
        if self._dispatch_task and not self._dispatch_task.done():
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass
        
        if self._timeout_check_task and not self._timeout_check_task.done():
            self._timeout_check_task.cancel()
            try:
                await self._timeout_check_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Job dispatcher stopped")
    
    async def _dispatch_loop(self):
        """Main dispatch loop."""
        while self._running:
            try:
                await self._dispatch_jobs()
                await asyncio.sleep(self.dispatch_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in dispatch loop: {str(e)}", exc_info=True)
                await asyncio.sleep(self.dispatch_interval * 2)  # Back off on error
    
    async def _timeout_check_loop(self):
        """Check for timed out jobs."""
        while self._running:
            try:
                await self._check_job_timeouts()
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in timeout check loop: {str(e)}", exc_info=True)
                await asyncio.sleep(60)
    
    async def _dispatch_jobs(self):
        """Dispatch jobs to available compute instances."""
        try:
            # Get available compute instances
            available_instances = await self._get_available_instances()
            
            if not available_instances:
                logger.debug("No available compute instances for job dispatch")
                return
            
            # Dispatch jobs based on priority and instance availability
            dispatched_count = 0
            
            for instance in available_instances:
                if dispatched_count >= self.max_concurrent_dispatchers:
                    break
                
                # Get next job from queue
                job = await self.queue_service.dequeue()
                
                if not job:
                    break  # No more jobs to dispatch
                
                # Dispatch job to instance
                success = await self._dispatch_job_to_instance(job, instance)
                
                if success:
                    dispatched_count += 1
                    logger.info(
                        f"Job dispatched successfully",
                        job_id=job.id,
                        metadata={
                            "instance_id": instance["id"],
                            "priority": job.priority,
                            "word_count": job.word_count
                        }
                    )
                else:
                    # Put job back in queue if dispatch failed
                    await self.queue_service.enqueue(job)
                    logger.warning(
                        f"Failed to dispatch job, returned to queue",
                        job_id=job.id,
                        metadata={
                            "instance_id": instance["id"],
                            "priority": job.priority
                        }
                    )
            
            if dispatched_count > 0:
                logger.debug(f"Dispatched {dispatched_count} jobs")
                
        except Exception as e:
            logger.error(f"Error dispatching jobs: {str(e)}", exc_info=True)
    
    async def _get_available_instances(self) -> List[Dict]:
        """Get list of available compute instances."""
        try:
            async with get_db_session() as session:
                instance_repo = BaseRepository(session, ComputeInstance)
                
                # Get running instances with available capacity
                instances = await instance_repo.find_by({
                    "status": "running"
                })
                
                available_instances = []
                
                for instance in instances:
                    # Check if instance has capacity for more jobs
                    if instance.active_jobs < instance.max_concurrent_jobs:
                        # Check if instance is healthy (recent heartbeat)
                        if instance.last_heartbeat and \
                           datetime.utcnow() - instance.last_heartbeat < timedelta(minutes=5):
                            
                            available_capacity = instance.max_concurrent_jobs - instance.active_jobs
                            
                            available_instances.append({
                                "id": instance.id,
                                "instance_type": instance.instance_type,
                                "available_capacity": available_capacity,
                                "gpu_utilization": float(instance.gpu_utilization),
                                "memory_usage": float(instance.memory_usage),
                                "active_jobs": instance.active_jobs
                            })
                
                # Sort by available capacity and resource utilization
                available_instances.sort(
                    key=lambda x: (x["available_capacity"], -x["gpu_utilization"])
                )
                
                return available_instances
                
        except Exception as e:
            logger.error(f"Error getting available instances: {str(e)}", exc_info=True)
            return []
    
    async def _dispatch_job_to_instance(self, job: TranslationJob, instance: Dict) -> bool:
        """Dispatch a specific job to a compute instance."""
        try:
            instance_id = instance["id"]
            
            # Update job status to processing
            async with get_db_session() as session:
                job_repo = JobRepository(session)
                
                success = await job_repo.update_job_status(
                    job.id,
                    JobStatus.PROCESSING,
                    progress=0.0,
                    compute_instance_id=instance_id
                )
                
                if not success:
                    logger.error(f"Failed to update job status to processing", job_id=job.id)
                    return False
                
                # Update instance active job count
                instance_repo = BaseRepository(session, ComputeInstance)
                compute_instance = await instance_repo.get_by_id(instance_id)
                
                if compute_instance:
                    compute_instance.active_jobs += 1
                    await instance_repo.update(compute_instance)
                
                await session.commit()
            
            # Add to dispatcher tracking
            dispatcher_id = f"{instance_id}:{job.id}"
            self.active_dispatchers.add(dispatcher_id)
            
            # Start job processing (this would typically send to a worker service)
            asyncio.create_task(self._process_job(job, instance_id))
            
            return True
            
        except Exception as e:
            logger.error(
                f"Error dispatching job to instance",
                job_id=job.id,
                metadata={
                    "instance_id": instance.get("id"),
                    "error": str(e)
                },
                exc_info=True
            )
            return False
    
    async def _process_job(self, job: TranslationJob, instance_id: str):
        """Process a job (placeholder for actual translation processing)."""
        try:
            dispatcher_id = f"{instance_id}:{job.id}"
            start_time = time.time()
            
            logger.info(
                f"Job processing started",
                job_id=job.id,
                instance_id=instance_id,
                metadata={
                    "source_language": job.source_language,
                    "target_language": job.target_language,
                    "word_count": job.word_count,
                    "priority": job.priority
                }
            )
            
            # Simulate translation processing
            # In a real implementation, this would call the translation engine
            processing_time = self._estimate_processing_time(job.word_count, job.priority)
            
            # Update progress periodically
            progress_updates = [25, 50, 75]
            for progress in progress_updates:
                await asyncio.sleep(processing_time / 4)
                
                async with get_db_session() as session:
                    job_repo = JobRepository(session)
                    await job_repo.update_job_status(job.id, JobStatus.PROCESSING, progress=progress)
                    await session.commit()
                
                logger.debug(f"Job progress updated", job_id=job.id, metadata={"progress": progress})
            
            # Complete the job
            await asyncio.sleep(processing_time / 4)  # Final processing time
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            async with get_db_session() as session:
                job_repo = JobRepository(session)
                
                # Update job to completed
                await job_repo.update_job_status(
                    job.id,
                    JobStatus.COMPLETED,
                    progress=100.0,
                    processing_time_ms=processing_time_ms
                )
                
                # Update instance active job count
                instance_repo = BaseRepository(session, ComputeInstance)
                compute_instance = await instance_repo.get_by_id(instance_id)
                
                if compute_instance and compute_instance.active_jobs > 0:
                    compute_instance.active_jobs -= 1
                    await instance_repo.update(compute_instance)
                
                await session.commit()
            
            # Send completion notification
            completed_job = await self._get_job_by_id(job.id)
            if completed_job:
                await self.notification_service.send_job_completion_webhook(completed_job)
            
            logger.info(
                f"Job processing completed",
                job_id=job.id,
                instance_id=instance_id,
                metrics={
                    "processing_time_ms": processing_time_ms,
                    "word_count": job.word_count,
                    "words_per_minute": (job.word_count / (processing_time_ms / 1000 / 60)) if processing_time_ms > 0 else 0
                }
            )
            
        except asyncio.CancelledError:
            # Handle job cancellation
            await self._handle_job_cancellation(job.id, instance_id)
            
        except Exception as e:
            # Handle job failure
            await self._handle_job_failure(job.id, instance_id, str(e))
            
        finally:
            # Remove from active dispatchers
            dispatcher_id = f"{instance_id}:{job.id}"
            self.active_dispatchers.discard(dispatcher_id)
    
    def _estimate_processing_time(self, word_count: int, priority: str) -> float:
        """Estimate processing time based on word count and priority."""
        # Base processing rate: 1500 words per minute
        base_rate = config.performance.target_words_per_minute
        
        # Priority multipliers (higher priority gets more resources)
        priority_multipliers = {
            Priority.CRITICAL.value: 1.5,
            Priority.HIGH.value: 1.2,
            Priority.NORMAL.value: 1.0
        }
        
        multiplier = priority_multipliers.get(priority, 1.0)
        effective_rate = base_rate * multiplier
        
        # Calculate time in seconds
        processing_time = (word_count / effective_rate) * 60
        
        # Add some randomness for simulation (±20%)
        import random
        processing_time *= random.uniform(0.8, 1.2)
        
        return max(1.0, processing_time)  # Minimum 1 second
    
    async def _handle_job_cancellation(self, job_id: UUID, instance_id: str):
        """Handle job cancellation."""
        try:
            async with get_db_session() as session:
                job_repo = JobRepository(session)
                
                await job_repo.update_job_status(
                    job_id,
                    JobStatus.FAILED,
                    error_message="Job cancelled during processing"
                )
                
                # Update instance active job count
                instance_repo = BaseRepository(session, ComputeInstance)
                compute_instance = await instance_repo.get_by_id(instance_id)
                
                if compute_instance and compute_instance.active_jobs > 0:
                    compute_instance.active_jobs -= 1
                    await instance_repo.update(compute_instance)
                
                await session.commit()
            
            logger.warning(
                f"Job cancelled during processing",
                job_id=job_id,
                instance_id=instance_id
            )
            
        except Exception as e:
            logger.error(f"Error handling job cancellation: {str(e)}", exc_info=True)
    
    async def _handle_job_failure(self, job_id: UUID, instance_id: str, error_message: str):
        """Handle job processing failure."""
        try:
            async with get_db_session() as session:
                job_repo = JobRepository(session)
                
                await job_repo.update_job_status(
                    job_id,
                    JobStatus.FAILED,
                    error_message=error_message
                )
                
                # Update instance active job count
                instance_repo = BaseRepository(session, ComputeInstance)
                compute_instance = await instance_repo.get_by_id(instance_id)
                
                if compute_instance and compute_instance.active_jobs > 0:
                    compute_instance.active_jobs -= 1
                    await instance_repo.update(compute_instance)
                
                await session.commit()
            
            # Move to dead letter queue
            await self.queue_service.move_to_dead_letter_queue(job_id, error_message)
            
            # Send failure notification
            failed_job = await self._get_job_by_id(job_id)
            if failed_job:
                await self.notification_service.send_job_completion_webhook(failed_job)
            
            logger.error(
                f"Job processing failed",
                job_id=job_id,
                instance_id=instance_id,
                metadata={"error_message": error_message}
            )
            
        except Exception as e:
            logger.error(f"Error handling job failure: {str(e)}", exc_info=True)
    
    async def _check_job_timeouts(self):
        """Check for jobs that have exceeded timeout limits."""
        try:
            async with get_db_session() as session:
                job_repo = JobRepository(session)
                
                # Get stale processing jobs
                stale_jobs = await job_repo.get_stale_jobs(timeout_minutes=self.job_timeout_minutes)
                
                for job in stale_jobs:
                    await self._handle_job_timeout(job)
                    
                if stale_jobs:
                    logger.warning(f"Handled {len(stale_jobs)} timed out jobs")
                    
        except Exception as e:
            logger.error(f"Error checking job timeouts: {str(e)}", exc_info=True)
    
    async def _handle_job_timeout(self, job: TranslationJob):
        """Handle a job that has timed out."""
        try:
            timeout_message = f"Job timed out after {self.job_timeout_minutes} minutes"
            
            async with get_db_session() as session:
                job_repo = JobRepository(session)
                
                await job_repo.update_job_status(
                    job.id,
                    JobStatus.FAILED,
                    error_message=timeout_message
                )
                
                # Update instance active job count if assigned
                if job.compute_instance_id:
                    instance_repo = BaseRepository(session, ComputeInstance)
                    compute_instance = await instance_repo.get_by_id(job.compute_instance_id)
                    
                    if compute_instance and compute_instance.active_jobs > 0:
                        compute_instance.active_jobs -= 1
                        await instance_repo.update(compute_instance)
                
                await session.commit()
            
            # Move to dead letter queue
            await self.queue_service.move_to_dead_letter_queue(job.id, timeout_message)
            
            # Send timeout notification
            await self.notification_service.send_job_completion_webhook(job)
            
            logger.warning(
                f"Job timed out and moved to DLQ",
                job_id=job.id,
                metadata={
                    "timeout_minutes": self.job_timeout_minutes,
                    "instance_id": job.compute_instance_id
                }
            )
            
        except Exception as e:
            logger.error(f"Error handling job timeout: {str(e)}", exc_info=True)
    
    async def _get_job_by_id(self, job_id: UUID) -> Optional[TranslationJob]:
        """Get job by ID."""
        try:
            async with get_db_session() as session:
                job_repo = JobRepository(session)
                return await job_repo.get_by_id(job_id)
        except Exception as e:
            logger.error(f"Error getting job by ID: {str(e)}", exc_info=True)
            return None
    
    async def cancel_job(self, job_id: UUID) -> bool:
        """Cancel a specific job."""
        try:
            # Remove from queue if still queued
            removed_from_queue = await self.queue_service.remove_job(job_id)
            
            # If job is currently processing, it will be handled by the cancellation logic
            dispatcher_id = None
            for active_id in self.active_dispatchers:
                if str(job_id) in active_id:
                    dispatcher_id = active_id
                    break
            
            if dispatcher_id:
                # Job is currently processing - the processing task will handle cancellation
                logger.info(f"Job cancellation requested for processing job", job_id=job_id)
            elif removed_from_queue:
                # Job was in queue and removed
                async with get_db_session() as session:
                    job_repo = JobRepository(session)
                    await job_repo.update_job_status(
                        job_id,
                        JobStatus.FAILED,
                        error_message="Job cancelled by user"
                    )
                    await session.commit()
                
                logger.info(f"Job cancelled from queue", job_id=job_id)
            else:
                logger.warning(f"Job not found in queue or processing", job_id=job_id)
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling job {job_id}: {str(e)}", exc_info=True)
            return False
    
    async def get_scheduler_status(self) -> Dict:
        """Get current scheduler status."""
        try:
            available_instances = await self._get_available_instances()
            
            return {
                "running": self._running,
                "active_dispatchers": len(self.active_dispatchers),
                "max_concurrent_dispatchers": self.max_concurrent_dispatchers,
                "dispatch_interval": self.dispatch_interval,
                "job_timeout_minutes": self.job_timeout_minutes,
                "available_instances": len(available_instances),
                "total_instance_capacity": sum(inst["available_capacity"] for inst in available_instances),
                "instance_details": available_instances
            }
            
        except Exception as e:
            logger.error(f"Error getting scheduler status: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    async def update_scheduler_config(self, config_updates: Dict):
        """Update scheduler configuration."""
        try:
            if "dispatch_interval" in config_updates:
                self.dispatch_interval = max(0.1, float(config_updates["dispatch_interval"]))
            
            if "max_concurrent_dispatchers" in config_updates:
                self.max_concurrent_dispatchers = max(1, int(config_updates["max_concurrent_dispatchers"]))
            
            if "job_timeout_minutes" in config_updates:
                self.job_timeout_minutes = max(5, int(config_updates["job_timeout_minutes"]))
            
            logger.info(
                f"Scheduler configuration updated",
                metadata={
                    "dispatch_interval": self.dispatch_interval,
                    "max_concurrent_dispatchers": self.max_concurrent_dispatchers,
                    "job_timeout_minutes": self.job_timeout_minutes
                }
            )
            
        except Exception as e:
            logger.error(f"Error updating scheduler config: {str(e)}", exc_info=True)
            raise
    
    async def close(self):
        """Close scheduler and cleanup resources."""
        await self.stop_dispatcher()
        await self.queue_service.close()
        await self.notification_service.close()
        
        logger.info("Job scheduler closed")