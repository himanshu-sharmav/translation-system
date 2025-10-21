"""
Repository for translation job operations.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from uuid import UUID

from sqlalchemy import and_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.config import JobStatus, Priority
from src.database.models import TranslationJob
from src.database.repositories.base import BaseRepository, RepositoryMixin
from src.models.interfaces import JobRepository as IJobRepository
from src.utils.exceptions import DatabaseError
from src.utils.logging import TranslationLogger

logger = TranslationLogger(__name__, "job-repository")


class JobRepository(BaseRepository[TranslationJob], RepositoryMixin, IJobRepository):
    """Repository for translation job operations."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, TranslationJob)
    
    async def get_by_status(self, status: JobStatus) -> List[TranslationJob]:
        """Get jobs by status."""
        try:
            filters = {'status': status.value}
            return await self.find_by(filters)
        except Exception as e:
            logger.error(f"Error getting jobs by status {status}: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to get jobs by status: {str(e)}")
    
    async def get_by_user(self, user_id: str, limit: int = 100) -> List[TranslationJob]:
        """Get jobs for a specific user."""
        try:
            filters = {'user_id': user_id}
            stmt = select(self.model_class).where(
                self.model_class.user_id == user_id
            ).order_by(desc(self.model_class.created_at)).limit(limit)
            
            result = await self.session.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            logger.error(f"Error getting jobs for user {user_id}: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to get jobs for user: {str(e)}")
    
    async def get_queue_depth_by_priority(self, priority: Priority) -> int:
        """Get queue depth for specific priority."""
        try:
            stmt = select(func.count()).where(
                and_(
                    self.model_class.status == JobStatus.QUEUED.value,
                    self.model_class.priority == priority.value
                )
            )
            result = await self.session.execute(stmt)
            return result.scalar() or 0
        except Exception as e:
            logger.error(f"Error getting queue depth for priority {priority}: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to get queue depth: {str(e)}")
    
    async def get_next_job_by_priority(self) -> Optional[TranslationJob]:
        """Get the next job to process based on priority and creation time."""
        try:
            # Order by priority (critical=1, high=2, normal=3) then by creation time
            stmt = select(self.model_class).where(
                self.model_class.status == JobStatus.QUEUED.value
            ).order_by(
                func.case(
                    (self.model_class.priority == Priority.CRITICAL.value, 1),
                    (self.model_class.priority == Priority.HIGH.value, 2),
                    (self.model_class.priority == Priority.NORMAL.value, 3),
                    else_=4
                ),
                self.model_class.created_at
            ).limit(1)
            
            result = await self.session.execute(stmt)
            return result.scalars().first()
        except Exception as e:
            logger.error(f"Error getting next job by priority: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to get next job: {str(e)}")
    
    async def get_jobs_by_priority_queue(self, priority: Priority, limit: int = 100) -> List[TranslationJob]:
        """Get jobs in a specific priority queue."""
        try:
            stmt = select(self.model_class).where(
                and_(
                    self.model_class.status == JobStatus.QUEUED.value,
                    self.model_class.priority == priority.value
                )
            ).order_by(self.model_class.created_at).limit(limit)
            
            result = await self.session.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            logger.error(f"Error getting jobs by priority queue {priority}: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to get jobs by priority queue: {str(e)}")
    
    async def get_processing_jobs(self, instance_id: Optional[str] = None) -> List[TranslationJob]:
        """Get currently processing jobs, optionally filtered by instance."""
        try:
            filters = {'status': JobStatus.PROCESSING.value}
            if instance_id:
                filters['compute_instance_id'] = instance_id
            
            return await self.find_by(filters)
        except Exception as e:
            logger.error(f"Error getting processing jobs: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to get processing jobs: {str(e)}")
    
    async def get_stale_jobs(self, timeout_minutes: int = 30) -> List[TranslationJob]:
        """Get jobs that have been processing for too long."""
        try:
            timeout_threshold = datetime.utcnow() - timedelta(minutes=timeout_minutes)
            
            stmt = select(self.model_class).where(
                and_(
                    self.model_class.status == JobStatus.PROCESSING.value,
                    self.model_class.started_at < timeout_threshold
                )
            )
            
            result = await self.session.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            logger.error(f"Error getting stale jobs: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to get stale jobs: {str(e)}")
    
    async def update_job_status(self, job_id: UUID, status: JobStatus, 
                              progress: Optional[float] = None,
                              error_message: Optional[str] = None,
                              processing_time_ms: Optional[int] = None,
                              compute_instance_id: Optional[str] = None) -> bool:
        """Update job status and related fields."""
        try:
            job = await self.get_by_id(job_id)
            if not job:
                return False
            
            # Update status
            job.status = status.value
            
            # Update progress if provided
            if progress is not None:
                job.progress = min(100.0, max(0.0, progress))
            
            # Update timestamps based on status
            now = datetime.utcnow()
            if status == JobStatus.PROCESSING and not job.started_at:
                job.started_at = now
            elif status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                job.completed_at = now
                if progress is None:
                    job.progress = 100.0 if status == JobStatus.COMPLETED else job.progress
            
            # Update other fields
            if error_message is not None:
                job.error_message = error_message
            
            if processing_time_ms is not None:
                job.processing_time_ms = processing_time_ms
            
            if compute_instance_id is not None:
                job.compute_instance_id = compute_instance_id
            
            await self.session.flush()
            return True
            
        except Exception as e:
            logger.error(f"Error updating job status for {job_id}: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to update job status: {str(e)}")
    
    async def get_job_statistics(self, user_id: Optional[str] = None, 
                               days: int = 30) -> Dict[str, int]:
        """Get job statistics for a user or system-wide."""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            base_query = select(self.model_class).where(
                self.model_class.created_at >= start_date
            )
            
            if user_id:
                base_query = base_query.where(self.model_class.user_id == user_id)
            
            # Get counts by status
            stats = {}
            for status in JobStatus:
                count_query = select(func.count()).select_from(
                    base_query.where(self.model_class.status == status.value).subquery()
                )
                result = await self.session.execute(count_query)
                stats[status.value] = result.scalar() or 0
            
            # Get total word count
            word_count_query = select(func.sum(self.model_class.word_count)).select_from(
                base_query.subquery()
            )
            result = await self.session.execute(word_count_query)
            stats['total_words'] = result.scalar() or 0
            
            # Get average processing time for completed jobs
            avg_time_query = select(func.avg(self.model_class.processing_time_ms)).select_from(
                base_query.where(
                    and_(
                        self.model_class.status == JobStatus.COMPLETED.value,
                        self.model_class.processing_time_ms.isnot(None)
                    )
                ).subquery()
            )
            result = await self.session.execute(avg_time_query)
            stats['avg_processing_time_ms'] = float(result.scalar() or 0)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting job statistics: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to get job statistics: {str(e)}")
    
    async def get_queue_position(self, job_id: UUID) -> Optional[int]:
        """Get the position of a job in its priority queue."""
        try:
            job = await self.get_by_id(job_id)
            if not job or job.status != JobStatus.QUEUED.value:
                return None
            
            # Count jobs with higher priority or same priority but earlier creation time
            stmt = select(func.count()).where(
                and_(
                    self.model_class.status == JobStatus.QUEUED.value,
                    func.case(
                        (self.model_class.priority == Priority.CRITICAL.value, 1),
                        (self.model_class.priority == Priority.HIGH.value, 2),
                        (self.model_class.priority == Priority.NORMAL.value, 3),
                        else_=4
                    ) < func.case(
                        (job.priority == Priority.CRITICAL.value, 1),
                        (job.priority == Priority.HIGH.value, 2),
                        (job.priority == Priority.NORMAL.value, 3),
                        else_=4
                    )
                )
            )
            
            # Add jobs with same priority but earlier creation time
            same_priority_stmt = select(func.count()).where(
                and_(
                    self.model_class.status == JobStatus.QUEUED.value,
                    self.model_class.priority == job.priority,
                    self.model_class.created_at < job.created_at
                )
            )
            
            higher_priority_result = await self.session.execute(stmt)
            same_priority_result = await self.session.execute(same_priority_stmt)
            
            higher_priority_count = higher_priority_result.scalar() or 0
            same_priority_count = same_priority_result.scalar() or 0
            
            return higher_priority_count + same_priority_count + 1
            
        except Exception as e:
            logger.error(f"Error getting queue position for job {job_id}: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to get queue position: {str(e)}")
    
    async def cleanup_old_jobs(self, days: int = 30, status_filter: Optional[JobStatus] = None) -> int:
        """Clean up old completed or failed jobs."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            filters = {
                'completed_at': {'lt': cutoff_date}
            }
            
            if status_filter:
                filters['status'] = status_filter.value
            else:
                # Only clean up completed or failed jobs by default
                filters['status'] = [JobStatus.COMPLETED.value, JobStatus.FAILED.value]
            
            deleted_count = await self.bulk_delete(filters)
            
            logger.info(f"Cleaned up {deleted_count} old jobs older than {days} days")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old jobs: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to cleanup old jobs: {str(e)}")
    
    async def get_jobs_by_content_hash(self, content_hash: str, 
                                     source_lang: str, target_lang: str) -> List[TranslationJob]:
        """Get jobs with the same content hash and language pair."""
        try:
            filters = {
                'content_hash': content_hash,
                'source_language': source_lang,
                'target_language': target_lang
            }
            return await self.find_by(filters)
        except Exception as e:
            logger.error(f"Error getting jobs by content hash: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to get jobs by content hash: {str(e)}")
    
    async def get_performance_metrics(self, hours: int = 24) -> Dict[str, float]:
        """Get performance metrics for the specified time period."""
        try:
            start_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Get completed jobs in the time period
            completed_jobs_query = select(self.model_class).where(
                and_(
                    self.model_class.status == JobStatus.COMPLETED.value,
                    self.model_class.completed_at >= start_time,
                    self.model_class.processing_time_ms.isnot(None)
                )
            )
            
            result = await self.session.execute(completed_jobs_query)
            completed_jobs = list(result.scalars().all())
            
            if not completed_jobs:
                return {
                    'total_jobs': 0,
                    'avg_processing_time_ms': 0.0,
                    'total_words_processed': 0,
                    'avg_words_per_minute': 0.0,
                    'throughput_jobs_per_hour': 0.0
                }
            
            # Calculate metrics
            total_jobs = len(completed_jobs)
            total_processing_time = sum(job.processing_time_ms for job in completed_jobs)
            total_words = sum(job.word_count for job in completed_jobs)
            
            avg_processing_time = total_processing_time / total_jobs
            avg_words_per_minute = (total_words / (total_processing_time / 1000 / 60)) if total_processing_time > 0 else 0
            throughput_jobs_per_hour = total_jobs / hours
            
            return {
                'total_jobs': total_jobs,
                'avg_processing_time_ms': avg_processing_time,
                'total_words_processed': total_words,
                'avg_words_per_minute': avg_words_per_minute,
                'throughput_jobs_per_hour': throughput_jobs_per_hour
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to get performance metrics: {str(e)}")