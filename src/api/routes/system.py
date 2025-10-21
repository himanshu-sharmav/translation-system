"""
System API routes for health checks, statistics, and admin functions.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from src.api.dependencies import (
    get_job_repository, get_cache_repository, get_metrics_repository,
    get_current_active_user, get_admin_user, get_optional_user
)
from src.api.models import (
    HealthCheckResponseModel, SystemStatsResponseModel, SupportedLanguagesResponseModel,
    CostEstimateRequestModel, CostEstimateResponseModel, ErrorResponseModel
)
from src.config.config import config
from src.database.connection import db_manager
from src.database.repositories import JobRepository, CacheRepository, MetricsRepository
from src.services.cost_calculator import CostCalculator
from src.utils.logging import api_logger

router = APIRouter(prefix="/api/v1", tags=["system"])

# Service start time for uptime calculation
SERVICE_START_TIME = time.time()


@router.get(
    "/health",
    response_model=HealthCheckResponseModel,
    responses={
        503: {"model": ErrorResponseModel, "description": "Service unavailable"}
    },
    summary="Health check",
    description="Check the health status of the translation service and its dependencies."
)
async def health_check(
    job_repo: JobRepository = Depends(get_job_repository),
    cache_repo: CacheRepository = Depends(get_cache_repository)
) -> HealthCheckResponseModel:
    """Perform health check of the service and its dependencies."""
    try:
        # Check database connectivity
        database_status = "healthy"
        try:
            db_healthy = await db_manager.health_check()
            if not db_healthy:
                database_status = "unhealthy"
        except Exception:
            database_status = "unhealthy"
        
        # Check cache connectivity (simplified - would check Redis in production)
        cache_status = "healthy"
        try:
            # Try to get cache statistics as a health check
            await cache_repo.get_cache_statistics(days=1)
        except Exception:
            cache_status = "unhealthy"
        
        # Check queue system (simplified)
        queue_status = "healthy"
        try:
            # Try to get queue depth as a health check
            await job_repo.get_queue_depth_by_priority("normal")
        except Exception:
            queue_status = "unhealthy"
        
        # Determine overall status
        overall_status = "healthy"
        if database_status != "healthy" or cache_status != "healthy" or queue_status != "healthy":
            overall_status = "degraded"
        
        # Calculate uptime
        uptime_seconds = time.time() - SERVICE_START_TIME
        
        health_response = HealthCheckResponseModel(
            status=overall_status,
            timestamp=datetime.utcnow(),
            version="1.0.0",
            database=database_status,
            cache=cache_status,
            queue=queue_status,
            uptime_seconds=uptime_seconds
        )
        
        # Return 503 if service is unhealthy
        if overall_status == "degraded":
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=health_response.dict()
            )
        
        return health_response
        
    except Exception as e:
        api_logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0.0",
                "database": "unknown",
                "cache": "unknown",
                "queue": "unknown",
                "uptime_seconds": time.time() - SERVICE_START_TIME,
                "error": "Health check failed"
            }
        )


@router.get(
    "/stats",
    response_model=SystemStatsResponseModel,
    responses={
        401: {"model": ErrorResponseModel, "description": "Authentication required"},
        500: {"model": ErrorResponseModel, "description": "Internal server error"}
    },
    summary="Get system statistics",
    description="Get current system statistics including job counts, performance metrics, and resource usage."
)
async def get_system_stats(
    current_user: dict = Depends(get_current_active_user),
    job_repo: JobRepository = Depends(get_job_repository),
    cache_repo: CacheRepository = Depends(get_cache_repository),
    metrics_repo: MetricsRepository = Depends(get_metrics_repository)
) -> SystemStatsResponseModel:
    """Get system statistics."""
    try:
        # Get job statistics
        job_stats = await job_repo.get_job_statistics(days=1)
        
        # Get queue depths by priority
        queue_depths = {}
        for priority in ["normal", "high", "critical"]:
            depth = await job_repo.get_queue_depth_by_priority(priority)
            queue_depths[priority] = depth
        
        # Get performance metrics
        performance_metrics = await job_repo.get_performance_metrics(hours=1)
        
        # Get cache statistics
        cache_stats = await cache_repo.get_cache_statistics(days=1)
        
        # Get system health metrics
        health_metrics = await metrics_repo.get_system_health_metrics(minutes=5)
        
        # Extract relevant metrics
        cache_hit_ratio = 0.0
        if cache_stats["total_accesses"] > 0:
            cache_hit_ratio = min(1.0, cache_stats["total_accesses"] / 
                                (cache_stats["total_accesses"] + job_stats.get("completed", 0)))
        
        # Get active instances count
        active_instances = 0
        try:
            active_instances_metric = await metrics_repo.get_latest_metric("active_instances_count")
            if active_instances_metric:
                active_instances = int(active_instances_metric.metric_value)
        except Exception:
            active_instances = 1  # Default assumption
        
        api_logger.info(
            f"System stats requested",
            metadata={
                "user_id": current_user["user_id"],
                "total_jobs": job_stats.get("completed", 0) + job_stats.get("failed", 0) + job_stats.get("queued", 0),
                "active_jobs": job_stats.get("queued", 0) + job_stats.get("processing", 0)
            }
        )
        
        return SystemStatsResponseModel(
            total_jobs=job_stats.get("completed", 0) + job_stats.get("failed", 0) + job_stats.get("queued", 0),
            active_jobs=job_stats.get("queued", 0) + job_stats.get("processing", 0),
            completed_jobs=job_stats.get("completed", 0),
            failed_jobs=job_stats.get("failed", 0),
            queue_depth=queue_depths,
            avg_processing_time_ms=job_stats.get("avg_processing_time_ms", 0.0),
            words_per_minute=performance_metrics.get("avg_words_per_minute", 0.0),
            cache_hit_ratio=cache_hit_ratio,
            active_instances=active_instances
        )
        
    except Exception as e:
        api_logger.error(f"Error getting system stats: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"code": "INTERNAL_ERROR", "message": "Failed to get system statistics"}}
        )


@router.get(
    "/languages",
    response_model=SupportedLanguagesResponseModel,
    summary="Get supported languages",
    description="Get list of supported languages and language pairs for translation."
)
async def get_supported_languages(
    current_user: Optional[dict] = Depends(get_optional_user)
) -> SupportedLanguagesResponseModel:
    """Get supported languages."""
    try:
        # In production, this would come from a service or database
        supported_languages = [
            {"code": "en", "name": "English"},
            {"code": "es", "name": "Spanish"},
            {"code": "fr", "name": "French"},
            {"code": "de", "name": "German"},
            {"code": "zh", "name": "Chinese"},
            {"code": "ja", "name": "Japanese"},
            {"code": "ko", "name": "Korean"},
            {"code": "pt", "name": "Portuguese"},
            {"code": "it", "name": "Italian"},
            {"code": "ru", "name": "Russian"}
        ]
        
        # Calculate total possible pairs (n * (n-1) for directed pairs)
        total_pairs = len(supported_languages) * (len(supported_languages) - 1)
        
        if current_user:
            api_logger.info(
                f"Supported languages requested",
                metadata={
                    "user_id": current_user["user_id"],
                    "total_languages": len(supported_languages)
                }
            )
        
        return SupportedLanguagesResponseModel(
            languages=supported_languages,
            total_pairs=total_pairs
        )
        
    except Exception as e:
        api_logger.error(f"Error getting supported languages: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"code": "INTERNAL_ERROR", "message": "Failed to get supported languages"}}
        )


@router.post(
    "/cost-estimate",
    response_model=CostEstimateResponseModel,
    responses={
        400: {"model": ErrorResponseModel, "description": "Invalid request"},
        401: {"model": ErrorResponseModel, "description": "Authentication required"},
        500: {"model": ErrorResponseModel, "description": "Internal server error"}
    },
    summary="Get cost estimate",
    description="Get cost estimate for expected translation volume."
)
async def get_cost_estimate(
    request: CostEstimateRequestModel,
    current_user: dict = Depends(get_current_active_user)
) -> CostEstimateResponseModel:
    """Get cost estimate for translation volume."""
    try:
        calculator = CostCalculator()
        
        # Calculate cost estimate
        estimate = await calculator.calculate_cost_estimate(
            words_per_day=request.words_per_day,
            priority_distribution=request.priority_distribution
        )
        
        api_logger.info(
            f"Cost estimate requested",
            metadata={
                "user_id": current_user["user_id"],
                "words_per_day": request.words_per_day,
                "estimated_daily_cost": estimate["daily_cost_usd"]
            }
        )
        
        return CostEstimateResponseModel(
            words_per_day=request.words_per_day,
            estimated_daily_cost_usd=estimate["daily_cost_usd"],
            estimated_monthly_cost_usd=estimate["monthly_cost_usd"],
            breakdown=estimate["breakdown"],
            assumptions=estimate["assumptions"]
        )
        
    except Exception as e:
        api_logger.error(f"Error calculating cost estimate: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"code": "INTERNAL_ERROR", "message": "Failed to calculate cost estimate"}}
        )


@router.get(
    "/admin/metrics",
    responses={
        401: {"model": ErrorResponseModel, "description": "Authentication required"},
        403: {"model": ErrorResponseModel, "description": "Admin access required"},
        500: {"model": ErrorResponseModel, "description": "Internal server error"}
    },
    summary="Get detailed system metrics (Admin only)",
    description="Get detailed system metrics and performance data. Requires admin privileges."
)
async def get_admin_metrics(
    hours: int = 24,
    admin_user: dict = Depends(get_admin_user),
    metrics_repo: MetricsRepository = Depends(get_metrics_repository)
) -> Dict:
    """Get detailed system metrics for administrators."""
    try:
        if hours < 1 or hours > 168:  # Max 1 week
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": {"code": "INVALID_PARAMETER", "message": "Hours must be between 1 and 168"}}
            )
        
        # Get system health metrics
        health_metrics = await metrics_repo.get_system_health_metrics(minutes=5)
        
        # Get capacity metrics
        capacity_metrics = await metrics_repo.get_capacity_metrics(hours=1)
        
        # Get performance trends
        performance_trends = {}
        key_metrics = [
            "translation_latency_seconds",
            "gpu_utilization_percent",
            "memory_usage_percent",
            "queue_depth_total"
        ]
        
        for metric_name in key_metrics:
            trends = await metrics_repo.get_performance_trends(
                metric_name, hours=hours, interval_minutes=60
            )
            performance_trends[metric_name] = trends
        
        # Get alert metrics
        alert_thresholds = {
            "gpu_utilization_percent": {"max": 90.0, "severity": "warning"},
            "memory_usage_percent": {"max": 85.0, "severity": "warning"},
            "translation_latency_seconds": {"max": 10.0, "severity": "critical"},
            "queue_depth_total": {"max": 100.0, "severity": "warning"}
        }
        
        alerts = await metrics_repo.get_alert_metrics(alert_thresholds)
        
        api_logger.info(
            f"Admin metrics requested",
            metadata={
                "admin_user_id": admin_user["user_id"],
                "hours": hours,
                "active_alerts": len(alerts)
            }
        )
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "time_window_hours": hours,
            "health_metrics": health_metrics,
            "capacity_metrics": capacity_metrics,
            "performance_trends": performance_trends,
            "active_alerts": alerts,
            "summary": {
                "total_alerts": len(alerts),
                "critical_alerts": len([a for a in alerts if a.get("severity") == "critical"]),
                "warning_alerts": len([a for a in alerts if a.get("severity") == "warning"])
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error getting admin metrics: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"code": "INTERNAL_ERROR", "message": "Failed to get admin metrics"}}
        )


@router.post(
    "/admin/cleanup",
    responses={
        401: {"model": ErrorResponseModel, "description": "Authentication required"},
        403: {"model": ErrorResponseModel, "description": "Admin access required"},
        500: {"model": ErrorResponseModel, "description": "Internal server error"}
    },
    summary="Cleanup old data (Admin only)",
    description="Cleanup old jobs, cache entries, and metrics. Requires admin privileges."
)
async def cleanup_old_data(
    days: int = 30,
    admin_user: dict = Depends(get_admin_user),
    job_repo: JobRepository = Depends(get_job_repository),
    cache_repo: CacheRepository = Depends(get_cache_repository),
    metrics_repo: MetricsRepository = Depends(get_metrics_repository)
) -> Dict:
    """Cleanup old data from the system."""
    try:
        if days < 1 or days > 365:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": {"code": "INVALID_PARAMETER", "message": "Days must be between 1 and 365"}}
            )
        
        cleanup_results = {}
        
        # Cleanup old completed/failed jobs
        deleted_jobs = await job_repo.cleanup_old_jobs(days=days)
        cleanup_results["deleted_jobs"] = deleted_jobs
        
        # Cleanup expired cache entries
        deleted_cache_entries = await cache_repo.cleanup_expired(ttl_hours=days * 24)
        cleanup_results["deleted_cache_entries"] = deleted_cache_entries
        
        # Cleanup old metrics
        deleted_metrics = await metrics_repo.cleanup_old_metrics(days=days)
        cleanup_results["deleted_metrics"] = deleted_metrics
        
        api_logger.info(
            f"Data cleanup performed",
            metadata={
                "admin_user_id": admin_user["user_id"],
                "days": days,
                "deleted_jobs": deleted_jobs,
                "deleted_cache_entries": deleted_cache_entries,
                "deleted_metrics": deleted_metrics
            }
        )
        
        return {
            "message": "Cleanup completed successfully",
            "cleanup_period_days": days,
            "results": cleanup_results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error during cleanup: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"code": "INTERNAL_ERROR", "message": "Failed to cleanup old data"}}
        )