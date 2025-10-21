"""
Pydantic models for API request/response validation.
"""

from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any
from uuid import UUID

from pydantic import BaseModel, Field, validator


class TranslationRequestModel(BaseModel):
    """Model for translation request."""
    
    source_language: str = Field(..., min_length=2, max_length=10, description="Source language code")
    target_language: str = Field(..., min_length=2, max_length=10, description="Target language code")
    content: str = Field(..., min_length=1, max_length=500000, description="Text content to translate")
    priority: str = Field(default="normal", description="Translation priority")
    callback_url: Optional[str] = Field(None, max_length=500, description="Optional webhook URL for completion notification")
    
    @validator('priority')
    def validate_priority(cls, v):
        if v not in ['normal', 'high', 'critical']:
            raise ValueError('Priority must be one of: normal, high, critical')
        return v
    
    @validator('content')
    def validate_content_length(cls, v):
        word_count = len(v.split())
        if word_count > 15000:
            raise ValueError('Content exceeds maximum word limit of 15,000 words')
        return v
    
    @validator('source_language', 'target_language')
    def validate_language_codes(cls, v):
        # Basic validation - in production, this would check against supported languages
        if not v.isalpha() or len(v) < 2:
            raise ValueError('Invalid language code format')
        return v.lower()


class TranslationResponseModel(BaseModel):
    """Model for translation request response."""
    
    job_id: UUID = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Current job status")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    queue_position: Optional[int] = Field(None, description="Position in queue")
    message: str = Field(default="Translation request accepted", description="Response message")


class JobStatusResponseModel(BaseModel):
    """Model for job status response."""
    
    job_id: UUID = Field(..., description="Job identifier")
    status: str = Field(..., description="Current job status")
    progress: float = Field(..., ge=0, le=100, description="Completion progress percentage")
    created_at: datetime = Field(..., description="Job creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Job start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Job completion timestamp")
    word_count: int = Field(..., ge=0, description="Number of words in source content")
    priority: str = Field(..., description="Job priority level")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    download_url: Optional[str] = Field(None, description="URL to download results")
    error_message: Optional[str] = Field(None, description="Error message if job failed")
    processing_time_ms: Optional[int] = Field(None, ge=0, description="Processing time in milliseconds")


class TranslationResultModel(BaseModel):
    """Model for translation result."""
    
    job_id: UUID = Field(..., description="Job identifier")
    translated_content: str = Field(..., description="Translated text content")
    source_language: str = Field(..., description="Source language code")
    target_language: str = Field(..., description="Target language code")
    confidence_score: float = Field(..., ge=0, le=1, description="Translation confidence score")
    model_version: str = Field(..., description="Translation model version used")
    word_count: int = Field(..., ge=0, description="Number of words translated")
    processing_time_ms: int = Field(..., ge=0, description="Processing time in milliseconds")


class ErrorResponseModel(BaseModel):
    """Model for error responses."""
    
    error: Dict[str, Any] = Field(..., description="Error details")
    
    class Config:
        schema_extra = {
            "example": {
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Invalid request parameters",
                    "details": {
                        "field": "source_language",
                        "issue": "Language not supported"
                    }
                }
            }
        }


class HealthCheckResponseModel(BaseModel):
    """Model for health check response."""
    
    status: str = Field(..., description="Service health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    database: str = Field(..., description="Database connection status")
    cache: str = Field(..., description="Cache connection status")
    queue: str = Field(..., description="Queue system status")
    uptime_seconds: float = Field(..., ge=0, description="Service uptime in seconds")


class JobListResponseModel(BaseModel):
    """Model for job list response with pagination."""
    
    jobs: List[JobStatusResponseModel] = Field(..., description="List of jobs")
    total: int = Field(..., ge=0, description="Total number of jobs")
    page: int = Field(..., ge=1, description="Current page number")
    per_page: int = Field(..., ge=1, le=100, description="Items per page")
    total_pages: int = Field(..., ge=0, description="Total number of pages")
    has_prev: bool = Field(..., description="Whether there is a previous page")
    has_next: bool = Field(..., description="Whether there is a next page")


class SystemStatsResponseModel(BaseModel):
    """Model for system statistics response."""
    
    total_jobs: int = Field(..., ge=0, description="Total number of jobs processed")
    active_jobs: int = Field(..., ge=0, description="Number of currently active jobs")
    completed_jobs: int = Field(..., ge=0, description="Number of completed jobs")
    failed_jobs: int = Field(..., ge=0, description="Number of failed jobs")
    queue_depth: Dict[str, int] = Field(..., description="Queue depth by priority")
    avg_processing_time_ms: float = Field(..., ge=0, description="Average processing time")
    words_per_minute: float = Field(..., ge=0, description="Current processing rate")
    cache_hit_ratio: float = Field(..., ge=0, le=1, description="Cache hit ratio")
    active_instances: int = Field(..., ge=0, description="Number of active compute instances")


class SupportedLanguagesResponseModel(BaseModel):
    """Model for supported languages response."""
    
    languages: List[Dict[str, str]] = Field(..., description="List of supported languages")
    total_pairs: int = Field(..., ge=0, description="Total number of supported language pairs")
    
    class Config:
        schema_extra = {
            "example": {
                "languages": [
                    {"code": "en", "name": "English"},
                    {"code": "es", "name": "Spanish"},
                    {"code": "fr", "name": "French"}
                ],
                "total_pairs": 6
            }
        }


class CostEstimateRequestModel(BaseModel):
    """Model for cost estimate request."""
    
    words_per_day: int = Field(..., ge=1, le=10000000, description="Expected words per day")
    priority_distribution: Optional[Dict[str, float]] = Field(
        default={"normal": 0.8, "high": 0.15, "critical": 0.05},
        description="Distribution of priorities"
    )
    
    @validator('priority_distribution')
    def validate_priority_distribution(cls, v):
        if v is not None:
            total = sum(v.values())
            if abs(total - 1.0) > 0.01:  # Allow small floating point errors
                raise ValueError('Priority distribution must sum to 1.0')
            
            valid_priorities = {'normal', 'high', 'critical'}
            if not set(v.keys()).issubset(valid_priorities):
                raise ValueError('Invalid priority levels in distribution')
        
        return v


class CostEstimateResponseModel(BaseModel):
    """Model for cost estimate response."""
    
    words_per_day: int = Field(..., description="Words per day estimate is based on")
    estimated_daily_cost_usd: float = Field(..., ge=0, description="Estimated daily cost in USD")
    estimated_monthly_cost_usd: float = Field(..., ge=0, description="Estimated monthly cost in USD")
    breakdown: Dict[str, float] = Field(..., description="Cost breakdown by component")
    assumptions: Dict[str, Any] = Field(..., description="Assumptions used in calculation")
    
    class Config:
        schema_extra = {
            "example": {
                "words_per_day": 100000,
                "estimated_daily_cost_usd": 10.54,
                "estimated_monthly_cost_usd": 316.20,
                "breakdown": {
                    "compute_cost": 10.52,
                    "storage_cost": 0.008,
                    "network_cost": 0.009
                },
                "assumptions": {
                    "avg_processing_time_ms": 4000,
                    "gpu_instance_cost_per_hour": 0.526,
                    "words_per_minute": 1500
                }
            }
        }


# Request/Response models for batch operations
class BatchTranslationRequestModel(BaseModel):
    """Model for batch translation request."""
    
    requests: List[TranslationRequestModel] = Field(
        ..., 
        min_items=1, 
        max_items=100, 
        description="List of translation requests"
    )
    batch_priority: str = Field(default="normal", description="Priority for entire batch")
    
    @validator('batch_priority')
    def validate_batch_priority(cls, v):
        if v not in ['normal', 'high', 'critical']:
            raise ValueError('Batch priority must be one of: normal, high, critical')
        return v


class BatchTranslationResponseModel(BaseModel):
    """Model for batch translation response."""
    
    batch_id: UUID = Field(..., description="Unique batch identifier")
    job_ids: List[UUID] = Field(..., description="List of individual job IDs")
    total_jobs: int = Field(..., ge=1, description="Total number of jobs in batch")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated batch completion time")
    status: str = Field(default="accepted", description="Batch status")


# Webhook notification models
class WebhookNotificationModel(BaseModel):
    """Model for webhook notifications."""
    
    job_id: UUID = Field(..., description="Job identifier")
    status: str = Field(..., description="Job status")
    completed_at: datetime = Field(..., description="Completion timestamp")
    result_url: Optional[str] = Field(None, description="URL to download results")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    processing_time_ms: Optional[int] = Field(None, description="Processing time")
    
    class Config:
        schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "completed",
                "completed_at": "2024-01-01T12:00:00Z",
                "result_url": "/api/v1/jobs/550e8400-e29b-41d4-a716-446655440000/result",
                "processing_time_ms": 2500
            }
        }