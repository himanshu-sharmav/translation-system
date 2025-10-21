"""
Translation API routes.
"""

import hashlib
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse

from src.api.dependencies import (
    get_job_repository, get_cache_repository, get_current_active_user,
    check_rate_limit, validate_language_pair, validate_content_length,
    validate_priority, get_pagination_params, PaginationParams
)
from src.api.models import (
    TranslationRequestModel, TranslationResponseModel, JobStatusResponseModel,
    TranslationResultModel, ErrorResponseModel, JobListResponseModel,
    BatchTranslationRequestModel, BatchTranslationResponseModel
)
from src.config.config import Priority, JobStatus
from src.database.models import TranslationJob
from src.database.repositories import JobRepository, CacheRepository
from src.services.queue_service import QueueService
from src.services.notification_service import NotificationService
from src.utils.exceptions import (
    ValidationError, JobNotFoundError, UnsupportedLanguageError,
    create_error_response
)
from src.utils.logging import api_logger

router = APIRouter(prefix="/api/v1", tags=["translation"])


@router.post(
    "/translate",
    response_model=TranslationResponseModel,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponseModel, "description": "Invalid request"},
        401: {"model": ErrorResponseModel, "description": "Authentication required"},
        429: {"model": ErrorResponseModel, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponseModel, "description": "Internal server error"}
    },
    summary="Submit translation request",
    description="Submit a new translation request and receive a job ID for tracking progress."
)
async def create_translation_job(
    request: TranslationRequestModel,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(check_rate_limit),
    job_repo: JobRepository = Depends(get_job_repository),
    cache_repo: CacheRepository = Depends(get_cache_repository)
) -> TranslationResponseModel:
    """Create a new translation job."""
    try:
        # Validate language pair
        source_lang, target_lang = await validate_language_pair(
            request.source_language, request.target_language
        )
        
        # Validate content
        content = validate_content_length(request.content)
        
        # Validate priority
        priority = validate_priority(request.priority)
        
        # Generate content hash for caching
        content_hash = hashlib.sha256(
            f"{source_lang}:{target_lang}:{content}".encode()
        ).hexdigest()
        
        # Check cache first
        cached_result = await cache_repo.get_by_hash(
            content_hash, source_lang, target_lang
        )
        
        if cached_result:
            # Update cache access stats
            await cache_repo.update_access_stats(cached_result.id)
            
            # Return cached result immediately
            api_logger.info(
                f"Cache hit for translation request",
                metadata={
                    "user_id": current_user["user_id"],
                    "source_language": source_lang,
                    "target_language": target_lang,
                    "cache_hit": True
                }
            )
            
            return TranslationResponseModel(
                job_id=uuid4(),  # Generate a dummy job ID for consistency
                status="completed",
                message="Translation retrieved from cache"
            )
        
        # Create new translation job
        job_id = uuid4()
        word_count = len(content.split())
        
        job = TranslationJob(
            id=job_id,
            user_id=current_user["user_id"],
            source_language=source_lang,
            target_language=target_lang,
            content_hash=content_hash,
            word_count=word_count,
            priority=priority,
            status=JobStatus.QUEUED.value,
            callback_url=request.callback_url
        )
        
        # Save job to database
        created_job = await job_repo.create(job)
        
        # Add job to queue
        queue_service = QueueService()
        await queue_service.enqueue_job(created_job)
        
        # Get queue position
        queue_position = await job_repo.get_queue_position(job_id)
        
        # Estimate completion time based on queue position and processing rate
        estimated_completion = None
        if queue_position:
            # Simple estimation: assume 1500 words per minute processing rate
            estimated_minutes = (queue_position * word_count) / 1500
            estimated_completion = datetime.utcnow().replace(
                microsecond=0
            ) + timedelta(minutes=estimated_minutes)
        
        api_logger.info(
            f"Translation job created",
            job_id=job_id,
            metadata={
                "user_id": current_user["user_id"],
                "source_language": source_lang,
                "target_language": target_lang,
                "word_count": word_count,
                "priority": priority,
                "queue_position": queue_position
            }
        )
        
        return TranslationResponseModel(
            job_id=job_id,
            status=JobStatus.QUEUED.value,
            estimated_completion=estimated_completion,
            queue_position=queue_position,
            message="Translation request accepted and queued for processing"
        )
        
    except HTTPException:
        raise
    except ValidationError as e:
        api_logger.warning(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=create_error_response(e)
        )
    except UnsupportedLanguageError as e:
        api_logger.warning(f"Unsupported language: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=create_error_response(e)
        )
    except Exception as e:
        api_logger.error(f"Error creating translation job: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"code": "INTERNAL_ERROR", "message": "Failed to create translation job"}}
        )


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponseModel,
    responses={
        404: {"model": ErrorResponseModel, "description": "Job not found"},
        401: {"model": ErrorResponseModel, "description": "Authentication required"},
        403: {"model": ErrorResponseModel, "description": "Access denied"},
        500: {"model": ErrorResponseModel, "description": "Internal server error"}
    },
    summary="Get job status",
    description="Get the current status and details of a translation job."
)
async def get_job_status(
    job_id: UUID,
    current_user: dict = Depends(get_current_active_user),
    job_repo: JobRepository = Depends(get_job_repository)
) -> JobStatusResponseModel:
    """Get translation job status."""
    try:
        # Get job from database
        job = await job_repo.get_by_id(job_id)
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": {"code": "JOB_NOT_FOUND", "message": f"Job {job_id} not found"}}
            )
        
        # Check if user has access to this job
        if job.user_id != current_user["user_id"] and not current_user.get("is_admin", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"error": {"code": "ACCESS_DENIED", "message": "You don't have access to this job"}}
            )
        
        # Build download URL if job is completed
        download_url = None
        if job.status == JobStatus.COMPLETED.value:
            download_url = f"/api/v1/jobs/{job_id}/result"
        
        # Get estimated completion time if still queued
        estimated_completion = job.estimated_completion
        if job.status == JobStatus.QUEUED.value and not estimated_completion:
            queue_position = await job_repo.get_queue_position(job_id)
            if queue_position:
                estimated_minutes = (queue_position * job.word_count) / 1500
                estimated_completion = datetime.utcnow().replace(
                    microsecond=0
                ) + timedelta(minutes=estimated_minutes)
        
        api_logger.info(
            f"Job status requested",
            job_id=job_id,
            metadata={
                "user_id": current_user["user_id"],
                "job_status": job.status,
                "progress": float(job.progress)
            }
        )
        
        return JobStatusResponseModel(
            job_id=job.id,
            status=job.status,
            progress=float(job.progress),
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            word_count=job.word_count,
            priority=job.priority,
            estimated_completion=estimated_completion,
            download_url=download_url,
            error_message=job.error_message,
            processing_time_ms=job.processing_time_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error getting job status for {job_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"code": "INTERNAL_ERROR", "message": "Failed to get job status"}}
        )


@router.get(
    "/jobs/{job_id}/result",
    response_model=TranslationResultModel,
    responses={
        404: {"model": ErrorResponseModel, "description": "Job not found or not completed"},
        401: {"model": ErrorResponseModel, "description": "Authentication required"},
        403: {"model": ErrorResponseModel, "description": "Access denied"},
        500: {"model": ErrorResponseModel, "description": "Internal server error"}
    },
    summary="Get translation result",
    description="Download the translated content for a completed job."
)
async def get_translation_result(
    job_id: UUID,
    current_user: dict = Depends(get_current_active_user),
    job_repo: JobRepository = Depends(get_job_repository),
    cache_repo: CacheRepository = Depends(get_cache_repository)
) -> TranslationResultModel:
    """Get translation result for completed job."""
    try:
        # Get job from database
        job = await job_repo.get_by_id(job_id)
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": {"code": "JOB_NOT_FOUND", "message": f"Job {job_id} not found"}}
            )
        
        # Check if user has access to this job
        if job.user_id != current_user["user_id"] and not current_user.get("is_admin", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"error": {"code": "ACCESS_DENIED", "message": "You don't have access to this job"}}
            )
        
        # Check if job is completed
        if job.status != JobStatus.COMPLETED.value:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "code": "RESULT_NOT_AVAILABLE",
                        "message": f"Translation result not available. Job status: {job.status}"
                    }
                }
            )
        
        # Get translation result from cache
        cached_result = await cache_repo.get_by_hash(
            job.content_hash, job.source_language, job.target_language
        )
        
        if not cached_result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "code": "RESULT_NOT_FOUND",
                        "message": "Translation result not found in cache"
                    }
                }
            )
        
        # Update cache access stats
        await cache_repo.update_access_stats(cached_result.id)
        
        api_logger.info(
            f"Translation result downloaded",
            job_id=job_id,
            metadata={
                "user_id": current_user["user_id"],
                "word_count": job.word_count,
                "processing_time_ms": job.processing_time_ms
            }
        )
        
        return TranslationResultModel(
            job_id=job.id,
            translated_content=cached_result.translated_content,
            source_language=job.source_language,
            target_language=job.target_language,
            confidence_score=float(cached_result.confidence_score or 0.0),
            model_version=cached_result.model_version,
            word_count=job.word_count,
            processing_time_ms=job.processing_time_ms or 0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error getting translation result for {job_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"code": "INTERNAL_ERROR", "message": "Failed to get translation result"}}
        )


@router.get(
    "/jobs",
    response_model=JobListResponseModel,
    responses={
        401: {"model": ErrorResponseModel, "description": "Authentication required"},
        500: {"model": ErrorResponseModel, "description": "Internal server error"}
    },
    summary="List user jobs",
    description="Get a paginated list of translation jobs for the current user."
)
async def list_user_jobs(
    current_user: dict = Depends(get_current_active_user),
    pagination: PaginationParams = Depends(get_pagination_params),
    status_filter: Optional[str] = None,
    priority_filter: Optional[str] = None,
    job_repo: JobRepository = Depends(get_job_repository)
) -> JobListResponseModel:
    """List translation jobs for current user."""
    try:
        # Build filters
        filters = {"user_id": current_user["user_id"]}
        
        if status_filter:
            if status_filter not in ["queued", "processing", "completed", "failed"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"error": {"code": "INVALID_STATUS", "message": f"Invalid status filter: {status_filter}"}}
                )
            filters["status"] = status_filter
        
        if priority_filter:
            if priority_filter not in ["normal", "high", "critical"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"error": {"code": "INVALID_PRIORITY", "message": f"Invalid priority filter: {priority_filter}"}}
                )
            filters["priority"] = priority_filter
        
        # Get paginated jobs
        result = await job_repo.paginate(
            page=pagination.page,
            per_page=pagination.per_page,
            filters=filters
        )
        
        # Convert jobs to response models
        job_responses = []
        for job in result["items"]:
            download_url = None
            if job.status == JobStatus.COMPLETED.value:
                download_url = f"/api/v1/jobs/{job.id}/result"
            
            job_responses.append(JobStatusResponseModel(
                job_id=job.id,
                status=job.status,
                progress=float(job.progress),
                created_at=job.created_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                word_count=job.word_count,
                priority=job.priority,
                estimated_completion=job.estimated_completion,
                download_url=download_url,
                error_message=job.error_message,
                processing_time_ms=job.processing_time_ms
            ))
        
        api_logger.info(
            f"Job list requested",
            metadata={
                "user_id": current_user["user_id"],
                "page": pagination.page,
                "per_page": pagination.per_page,
                "total_jobs": result["total"],
                "status_filter": status_filter,
                "priority_filter": priority_filter
            }
        )
        
        return JobListResponseModel(
            jobs=job_responses,
            total=result["total"],
            page=result["page"],
            per_page=result["per_page"],
            total_pages=result["total_pages"],
            has_prev=result["has_prev"],
            has_next=result["has_next"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error listing jobs for user {current_user['user_id']}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"code": "INTERNAL_ERROR", "message": "Failed to list jobs"}}
        )


@router.delete(
    "/jobs/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: {"model": ErrorResponseModel, "description": "Job not found"},
        401: {"model": ErrorResponseModel, "description": "Authentication required"},
        403: {"model": ErrorResponseModel, "description": "Access denied"},
        409: {"model": ErrorResponseModel, "description": "Job cannot be cancelled"},
        500: {"model": ErrorResponseModel, "description": "Internal server error"}
    },
    summary="Cancel translation job",
    description="Cancel a queued or processing translation job."
)
async def cancel_translation_job(
    job_id: UUID,
    current_user: dict = Depends(get_current_active_user),
    job_repo: JobRepository = Depends(get_job_repository)
):
    """Cancel a translation job."""
    try:
        # Get job from database
        job = await job_repo.get_by_id(job_id)
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": {"code": "JOB_NOT_FOUND", "message": f"Job {job_id} not found"}}
            )
        
        # Check if user has access to this job
        if job.user_id != current_user["user_id"] and not current_user.get("is_admin", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"error": {"code": "ACCESS_DENIED", "message": "You don't have access to this job"}}
            )
        
        # Check if job can be cancelled
        if job.status in [JobStatus.COMPLETED.value, JobStatus.FAILED.value]:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error": {
                        "code": "CANNOT_CANCEL",
                        "message": f"Cannot cancel job with status: {job.status}"
                    }
                }
            )
        
        # Update job status to failed (cancelled)
        await job_repo.update_job_status(
            job_id,
            JobStatus.FAILED,
            error_message="Job cancelled by user"
        )
        
        # Remove from queue if still queued
        if job.status == JobStatus.QUEUED.value:
            queue_service = QueueService()
            await queue_service.remove_job(job_id)
        
        api_logger.info(
            f"Translation job cancelled",
            job_id=job_id,
            metadata={
                "user_id": current_user["user_id"],
                "original_status": job.status
            }
        )
        
        return JSONResponse(
            status_code=status.HTTP_204_NO_CONTENT,
            content=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error cancelling job {job_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"code": "INTERNAL_ERROR", "message": "Failed to cancel job"}}
        )


@router.post(
    "/translate/batch",
    response_model=BatchTranslationResponseModel,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponseModel, "description": "Invalid request"},
        401: {"model": ErrorResponseModel, "description": "Authentication required"},
        429: {"model": ErrorResponseModel, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponseModel, "description": "Internal server error"}
    },
    summary="Submit batch translation request",
    description="Submit multiple translation requests as a batch."
)
async def create_batch_translation(
    request: BatchTranslationRequestModel,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(check_rate_limit),
    job_repo: JobRepository = Depends(get_job_repository)
) -> BatchTranslationResponseModel:
    """Create a batch of translation jobs."""
    try:
        batch_id = uuid4()
        job_ids = []
        
        # Process each translation request in the batch
        for translation_request in request.requests:
            # Validate language pair
            source_lang, target_lang = await validate_language_pair(
                translation_request.source_language, translation_request.target_language
            )
            
            # Validate content
            content = validate_content_length(translation_request.content)
            
            # Use batch priority if individual priority not specified
            priority = translation_request.priority or request.batch_priority
            priority = validate_priority(priority)
            
            # Generate content hash
            content_hash = hashlib.sha256(
                f"{source_lang}:{target_lang}:{content}".encode()
            ).hexdigest()
            
            # Create job
            job_id = uuid4()
            word_count = len(content.split())
            
            job = TranslationJob(
                id=job_id,
                user_id=current_user["user_id"],
                source_language=source_lang,
                target_language=target_lang,
                content_hash=content_hash,
                word_count=word_count,
                priority=priority,
                status=JobStatus.QUEUED.value,
                callback_url=translation_request.callback_url
            )
            
            # Save job to database
            created_job = await job_repo.create(job)
            job_ids.append(job_id)
            
            # Add job to queue
            queue_service = QueueService()
            await queue_service.enqueue_job(created_job)
        
        # Estimate batch completion time
        total_words = sum(len(req.content.split()) for req in request.requests)
        estimated_minutes = total_words / 1500  # Assume 1500 words per minute
        estimated_completion = datetime.utcnow().replace(
            microsecond=0
        ) + timedelta(minutes=estimated_minutes)
        
        api_logger.info(
            f"Batch translation created",
            metadata={
                "user_id": current_user["user_id"],
                "batch_id": str(batch_id),
                "total_jobs": len(job_ids),
                "total_words": total_words,
                "batch_priority": request.batch_priority
            }
        )
        
        return BatchTranslationResponseModel(
            batch_id=batch_id,
            job_ids=job_ids,
            total_jobs=len(job_ids),
            estimated_completion=estimated_completion,
            status="accepted"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error creating batch translation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"code": "INTERNAL_ERROR", "message": "Failed to create batch translation"}}
        )