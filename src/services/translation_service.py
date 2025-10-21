"""
Translation service that orchestrates translation jobs using the translation engine.
"""

import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from src.config.config import config, JobStatus
from src.database.connection import get_db_session
from src.database.repositories import JobRepository, CacheRepository
from src.models.interfaces import TranslationRequest, TranslationJob, TranslationResult, CacheEntry
from src.services.translation_engine import create_translation_engine
from src.services.queue_service import QueueService
from src.utils.exceptions import (
    TranslationError, 
    UnsupportedLanguageError, 
    JobNotFoundError,
    ValidationError
)
from src.utils.logging import TranslationLogger

logger = TranslationLogger(__name__, "translation-service")


class TranslationService:
    """Service for managing translation jobs and orchestrating the translation process."""
    
    def __init__(self):
        self.translation_engine = create_translation_engine()
        self.queue_service = QueueService()
        self._processing_jobs: Dict[str, asyncio.Task] = {}
        self._shutdown = False
        
    async def submit_translation_job(self, request: TranslationRequest, user_id: str) -> TranslationJob:
        """Submit a new translation job."""
        try:
            # Validate request
            await self._validate_translation_request(request)
            
            # Generate content hash for caching
            content_hash = self._generate_content_hash(
                request.content, 
                request.source_language, 
                request.target_language
            )
            
            # Check cache first
            cached_result = await self._check_cache(
                content_hash, 
                request.source_language, 
                request.target_language
            )
            
            if cached_result:
                logger.info(
                    f"Cache hit for translation request",
                    metadata={
                        "content_hash": content_hash,
                        "source_language": request.source_language,
                        "target_language": request.target_language,
                        "user_id": user_id
                    }
                )
                
                # Create completed job from cache
                job = await self._create_job_from_cache(request, user_id, cached_result)
                return job
            
            # Create new translation job
            job = TranslationJob(
                id=uuid4(),
                user_id=user_id,
                source_language=request.source_language,
                target_language=request.target_language,
                content_hash=content_hash,
                word_count=len(request.content.split()),
                priority=request.priority,
                status=JobStatus.QUEUED,
                created_at=datetime.utcnow(),
                callback_url=request.callback_url,
                estimated_completion=self._estimate_completion_time(request)
            )
            
            # Store job in database
            async with get_db_session() as session:
                job_repo = JobRepository(session)
                created_job = await job_repo.create(job)
                await session.commit()
            
            # Add to queue
            await self.queue_service.enqueue(created_job)
            
            logger.info(
                f"Translation job submitted",
                job_id=created_job.id,
                metadata={
                    "user_id": user_id,
                    "source_language": request.source_language,
                    "target_language": request.target_language,
                    "word_count": job.word_count,
                    "priority": request.priority.value
                }
            )
            
            return created_job
            
        except Exception as e:
            logger.error(f"Failed to submit translation job: {str(e)}", exc_info=True)
            raise TranslationError(f"Failed to submit translation job: {str(e)}")
    
    async def get_job_status(self, job_id: UUID, user_id: str) -> Optional[TranslationJob]:
        """Get the status of a translation job."""
        try:
            async with get_db_session() as session:
                job_repo = JobRepository(session)
                job = await job_repo.get_by_id(job_id)
                
                if not job:
                    raise JobNotFoundError(str(job_id))
                
                # Check if user has access to this job
                if job.user_id != user_id:
                    raise JobNotFoundError(str(job_id))  # Don't reveal existence
                
                return job
                
        except JobNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to get job status for {job_id}: {str(e)}", exc_info=True)
            raise TranslationError(f"Failed to get job status: {str(e)}")
    
    async def get_translation_result(self, job_id: UUID, user_id: str) -> Optional[TranslationResult]:
        """Get the translation result for a completed job."""
        try:
            job = await self.get_job_status(job_id, user_id)
            
            if job.status != JobStatus.COMPLETED:
                return None
            
            # Get result from cache
            if job.content_hash:
                cached_result = await self._check_cache(
                    job.content_hash,
                    job.source_language,
                    job.target_language
                )
                
                if cached_result:
                    return TranslationResult(
                        job_id=job.id,
                        translated_content=cached_result.translated_content,
                        source_language=job.source_language,
                        target_language=job.target_language,
                        confidence_score=cached_result.confidence_score,
                        model_version=cached_result.model_version,
                        processing_time_ms=job.processing_time_ms or 0
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get translation result for {job_id}: {str(e)}", exc_info=True)
            raise TranslationError(f"Failed to get translation result: {str(e)}")
    
    async def cancel_job(self, job_id: UUID, user_id: str) -> bool:
        """Cancel a translation job."""
        try:
            job = await self.get_job_status(job_id, user_id)
            
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                return False  # Cannot cancel completed jobs
            
            # Remove from queue if still queued
            if job.status == JobStatus.QUEUED:
                await self.queue_service.remove_job(job_id)
            
            # Update job status
            async with get_db_session() as session:
                job_repo = JobRepository(session)
                await job_repo.update_job_status(
                    job_id,
                    JobStatus.FAILED,
                    error_message="Job cancelled by user"
                )
                await session.commit()
            
            logger.info(f"Translation job cancelled", job_id=job_id, user_id=user_id)
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {str(e)}", exc_info=True)
            raise TranslationError(f"Failed to cancel job: {str(e)}")
    
    async def process_translation_job(self, job: TranslationJob) -> TranslationResult:
        """Process a translation job using the translation engine."""
        try:
            logger.info(f"Processing translation job", job_id=job.id)
            
            # Update job status to processing
            async with get_db_session() as session:
                job_repo = JobRepository(session)
                await job_repo.update_job_status(job.id, JobStatus.PROCESSING, progress=0.0)
                await session.commit()
            
            # Get source content from cache or database
            source_content = await self._get_source_content(job)
            
            if not source_content:
                raise TranslationError("Source content not found")
            
            # Create translation request
            request = TranslationRequest(
                source_language=job.source_language,
                target_language=job.target_language,
                content=source_content,
                priority=job.priority,
                user_id=job.user_id
            )
            
            # Update progress
            async with get_db_session() as session:
                job_repo = JobRepository(session)
                await job_repo.update_job_status(job.id, JobStatus.PROCESSING, progress=25.0)
                await session.commit()
            
            # Perform translation
            result = await self.translation_engine.translate(request)
            result.job_id = job.id
            
            # Update progress
            async with get_db_session() as session:
                job_repo = JobRepository(session)
                await job_repo.update_job_status(job.id, JobStatus.PROCESSING, progress=75.0)
                await session.commit()
            
            # Cache the result
            await self._cache_translation_result(job, result, source_content)
            
            # Update job to completed
            async with get_db_session() as session:
                job_repo = JobRepository(session)
                await job_repo.update_job_status(
                    job.id,
                    JobStatus.COMPLETED,
                    progress=100.0,
                    processing_time_ms=result.processing_time_ms
                )
                await session.commit()
            
            logger.info(
                f"Translation job completed",
                job_id=job.id,
                metrics={
                    "processing_time_ms": result.processing_time_ms,
                    "word_count": job.word_count,
                    "confidence_score": result.confidence_score
                }
            )
            
            return result
            
        except Exception as e:
            # Update job to failed
            async with get_db_session() as session:
                job_repo = JobRepository(session)
                await job_repo.update_job_status(
                    job.id,
                    JobStatus.FAILED,
                    error_message=str(e)
                )
                await session.commit()
            
            logger.error(f"Translation job failed", job_id=job.id, error=str(e), exc_info=True)
            raise TranslationError(f"Translation job failed: {str(e)}")
    
    async def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return self.translation_engine.get_supported_languages()
    
    async def get_engine_health(self) -> Dict:
        """Get translation engine health status."""
        if hasattr(self.translation_engine, 'health_check'):
            return await self.translation_engine.health_check()
        else:
            return {
                "status": "healthy",
                "supported_languages": len(self.translation_engine.get_supported_languages()),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _validate_translation_request(self, request: TranslationRequest) -> None:
        """Validate translation request."""
        if not request.content or not request.content.strip():
            raise ValidationError("Content cannot be empty")
        
        if len(request.content.split()) > config.performance.max_document_words:
            raise ValidationError(
                f"Content exceeds maximum word limit of {config.performance.max_document_words} words"
            )
        
        supported_languages = self.translation_engine.get_supported_languages()
        
        if request.source_language not in supported_languages:
            raise UnsupportedLanguageError(
                request.source_language, 
                request.target_language, 
                supported_languages
            )
        
        if request.target_language not in supported_languages:
            raise UnsupportedLanguageError(
                request.source_language, 
                request.target_language, 
                supported_languages
            )
        
        if request.source_language == request.target_language:
            raise ValidationError("Source and target languages cannot be the same")
    
    def _generate_content_hash(self, content: str, source_lang: str, target_lang: str) -> str:
        """Generate hash for content caching."""
        content_str = f"{source_lang}:{target_lang}:{content.strip()}"
        return hashlib.sha256(content_str.encode('utf-8')).hexdigest()
    
    async def _check_cache(self, content_hash: str, source_lang: str, target_lang: str) -> Optional[CacheEntry]:
        """Check if translation exists in cache."""
        try:
            async with get_db_session() as session:
                cache_repo = CacheRepository(session)
                return await cache_repo.get_by_hash(content_hash, source_lang, target_lang)
        except Exception as e:
            logger.warning(f"Cache lookup failed: {str(e)}")
            return None
    
    async def _create_job_from_cache(self, request: TranslationRequest, user_id: str, cached_result: CacheEntry) -> TranslationJob:
        """Create a completed job from cached result."""
        job = TranslationJob(
            id=uuid4(),
            user_id=user_id,
            source_language=request.source_language,
            target_language=request.target_language,
            content_hash=cached_result.content_hash,
            word_count=len(request.content.split()),
            priority=request.priority,
            status=JobStatus.COMPLETED,
            progress=100.0,
            created_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            callback_url=request.callback_url,
            processing_time_ms=100  # Minimal time for cache hit
        )
        
        # Store job in database
        async with get_db_session() as session:
            job_repo = JobRepository(session)
            created_job = await job_repo.create(job)
            
            # Update cache access count
            cached_result.access_count += 1
            cached_result.last_accessed = datetime.utcnow()
            await cache_repo.update(cached_result)
            
            await session.commit()
        
        return created_job
    
    def _estimate_completion_time(self, request: TranslationRequest) -> datetime:
        """Estimate job completion time."""
        word_count = len(request.content.split())
        
        # Base processing time calculation
        base_time_minutes = word_count / config.performance.target_words_per_minute
        
        # Priority multipliers
        priority_multipliers = {
            "critical": 1.0,
            "high": 1.5,
            "normal": 2.0
        }
        
        multiplier = priority_multipliers.get(request.priority.value, 2.0)
        estimated_minutes = base_time_minutes * multiplier
        
        # Add queue wait time estimate (simplified)
        queue_wait_minutes = 2.0  # This would be calculated based on current queue depth
        
        total_minutes = estimated_minutes + queue_wait_minutes
        return datetime.utcnow() + timedelta(minutes=total_minutes)
    
    async def _get_source_content(self, job: TranslationJob) -> Optional[str]:
        """Get source content for translation job."""
        # In a real implementation, this would retrieve the source content
        # from the database or cache based on the job's content_hash
        # For now, we'll return a placeholder
        return f"Source content for job {job.id}"
    
    async def _cache_translation_result(self, job: TranslationJob, result: TranslationResult, source_content: str) -> None:
        """Cache translation result."""
        try:
            cache_entry = CacheEntry(
                id=uuid4(),
                content_hash=job.content_hash,
                source_language=job.source_language,
                target_language=job.target_language,
                source_content=source_content,
                translated_content=result.translated_content,
                model_version=result.model_version,
                confidence_score=result.confidence_score,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow()
            )
            
            async with get_db_session() as session:
                cache_repo = CacheRepository(session)
                await cache_repo.create(cache_entry)
                await session.commit()
            
            logger.debug(f"Translation result cached", job_id=job.id, content_hash=job.content_hash)
            
        except Exception as e:
            logger.warning(f"Failed to cache translation result: {str(e)}")
            # Don't fail the job if caching fails
    
    async def close(self) -> None:
        """Clean up resources."""
        self._shutdown = True
        
        # Cancel any running processing tasks
        for task in self._processing_jobs.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self._processing_jobs:
            await asyncio.gather(*self._processing_jobs.values(), return_exceptions=True)
        
        # Close translation engine
        if hasattr(self.translation_engine, 'close'):
            await self.translation_engine.close()
        
        # Close queue service
        await self.queue_service.close()
        
        logger.info("Translation service shutdown complete")