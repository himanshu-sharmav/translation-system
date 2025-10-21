"""
Core interfaces and abstract base classes for the machine translation system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from uuid import UUID

from src.config.config import Priority, JobStatus


@dataclass
class TranslationRequest:
    """Request model for translation jobs."""
    source_language: str
    target_language: str
    content: str
    priority: Priority = Priority.NORMAL
    callback_url: Optional[str] = None
    user_id: str = ""
    
    def __post_init__(self):
        if len(self.content.split()) > 15000:
            raise ValueError("Content exceeds maximum word limit of 15,000 words")


@dataclass
class TranslationJob:
    """Model representing a translation job."""
    id: UUID
    user_id: str
    source_language: str
    target_language: str
    content_hash: str
    word_count: int
    priority: Priority
    status: JobStatus
    progress: float = 0.0
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    callback_url: Optional[str] = None
    result_url: Optional[str] = None
    error_message: Optional[str] = None
    compute_instance_id: Optional[str] = None
    processing_time_ms: Optional[int] = None


@dataclass
class TranslationResult:
    """Model representing translation results."""
    job_id: UUID
    translated_content: str
    source_language: str
    target_language: str
    confidence_score: float
    model_version: str
    processing_time_ms: int


@dataclass
class CacheEntry:
    """Model for cached translations."""
    id: UUID
    content_hash: str
    source_language: str
    target_language: str
    source_content: str
    translated_content: str
    model_version: str
    confidence_score: float
    access_count: int = 1
    last_accessed: datetime = None
    created_at: datetime = None


@dataclass
class SystemMetric:
    """Model for system performance metrics."""
    timestamp: datetime
    metric_name: str
    metric_value: float
    instance_id: Optional[str] = None
    tags: Optional[Dict[str, Any]] = None


@dataclass
class ComputeInstance:
    """Model representing a compute instance."""
    id: str
    instance_type: str
    status: str
    gpu_utilization: float
    memory_usage: float
    active_jobs: int
    created_at: datetime
    last_heartbeat: datetime


class TranslationEngine(ABC):
    """Abstract interface for translation engines."""
    
    @abstractmethod
    async def translate(self, request: TranslationRequest) -> TranslationResult:
        """Translate text from source to target language."""
        pass
    
    @abstractmethod
    async def load_model(self, model_id: str) -> bool:
        """Load a specific translation model."""
        pass
    
    @abstractmethod
    async def unload_model(self, model_id: str) -> bool:
        """Unload a translation model to free memory."""
        pass
    
    @abstractmethod
    def get_memory_usage(self) -> float:
        """Get current memory usage percentage."""
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        pass


class QueueManager(ABC):
    """Abstract interface for queue management."""
    
    @abstractmethod
    async def enqueue(self, job: TranslationJob) -> bool:
        """Add a job to the appropriate priority queue."""
        pass
    
    @abstractmethod
    async def dequeue(self, priority: Optional[Priority] = None) -> Optional[TranslationJob]:
        """Get the next job from the queue."""
        pass
    
    @abstractmethod
    async def get_queue_depth(self, priority: Optional[Priority] = None) -> int:
        """Get the number of jobs in queue."""
        pass
    
    @abstractmethod
    async def update_job_status(self, job_id: UUID, status: JobStatus, progress: float = None) -> bool:
        """Update job status and progress."""
        pass
    
    @abstractmethod
    async def get_job(self, job_id: UUID) -> Optional[TranslationJob]:
        """Get job details by ID."""
        pass


class CacheManager(ABC):
    """Abstract interface for caching system."""
    
    @abstractmethod
    async def get(self, cache_key: str) -> Optional[CacheEntry]:
        """Get cached translation result."""
        pass
    
    @abstractmethod
    async def set(self, cache_key: str, entry: CacheEntry, ttl: int = None) -> bool:
        """Store translation result in cache."""
        pass
    
    @abstractmethod
    async def invalidate(self, cache_key: str) -> bool:
        """Remove entry from cache."""
        pass
    
    @abstractmethod
    async def generate_cache_key(self, source_lang: str, target_lang: str, content: str, model_version: str) -> str:
        """Generate cache key for translation request."""
        pass
    
    @abstractmethod
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        pass


class ResourceManager(ABC):
    """Abstract interface for compute resource management."""
    
    @abstractmethod
    async def scale_up(self, instance_count: int = 1) -> List[str]:
        """Add new compute instances."""
        pass
    
    @abstractmethod
    async def scale_down(self, instance_count: int = 1) -> List[str]:
        """Remove compute instances."""
        pass
    
    @abstractmethod
    async def get_instance_metrics(self, instance_id: str) -> Dict[str, float]:
        """Get metrics for a specific instance."""
        pass
    
    @abstractmethod
    async def get_all_instances(self) -> List[ComputeInstance]:
        """Get all active compute instances."""
        pass
    
    @abstractmethod
    async def health_check(self, instance_id: str) -> bool:
        """Check if instance is healthy."""
        pass


class Repository(ABC):
    """Abstract base class for data repositories."""
    
    @abstractmethod
    async def create(self, entity: Any) -> Any:
        """Create a new entity."""
        pass
    
    @abstractmethod
    async def get_by_id(self, entity_id: Union[UUID, str]) -> Optional[Any]:
        """Get entity by ID."""
        pass
    
    @abstractmethod
    async def update(self, entity: Any) -> bool:
        """Update existing entity."""
        pass
    
    @abstractmethod
    async def delete(self, entity_id: Union[UUID, str]) -> bool:
        """Delete entity by ID."""
        pass


class JobRepository(Repository):
    """Repository interface for translation jobs."""
    
    @abstractmethod
    async def get_by_status(self, status: JobStatus) -> List[TranslationJob]:
        """Get jobs by status."""
        pass
    
    @abstractmethod
    async def get_by_user(self, user_id: str, limit: int = 100) -> List[TranslationJob]:
        """Get jobs for a specific user."""
        pass
    
    @abstractmethod
    async def get_queue_depth_by_priority(self, priority: Priority) -> int:
        """Get queue depth for specific priority."""
        pass


class CacheRepository(Repository):
    """Repository interface for cache entries."""
    
    @abstractmethod
    async def get_by_hash(self, content_hash: str, source_lang: str, target_lang: str) -> Optional[CacheEntry]:
        """Get cache entry by content hash and languages."""
        pass
    
    @abstractmethod
    async def cleanup_expired(self, ttl_hours: int) -> int:
        """Remove expired cache entries."""
        pass


class MetricsRepository(Repository):
    """Repository interface for system metrics."""
    
    @abstractmethod
    async def get_metrics(self, metric_name: str, start_time: datetime, end_time: datetime) -> List[SystemMetric]:
        """Get metrics within time range."""
        pass
    
    @abstractmethod
    async def get_latest_metric(self, metric_name: str, instance_id: str = None) -> Optional[SystemMetric]:
        """Get latest metric value."""
        pass


class MonitoringService(ABC):
    """Abstract interface for monitoring and alerting."""
    
    @abstractmethod
    async def record_metric(self, metric: SystemMetric) -> bool:
        """Record a system metric."""
        pass
    
    @abstractmethod
    async def check_alerts(self) -> List[Dict[str, Any]]:
        """Check for alert conditions."""
        pass
    
    @abstractmethod
    async def send_alert(self, alert: Dict[str, Any]) -> bool:
        """Send alert notification."""
        pass
    
    @abstractmethod
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        pass


class AuthenticationService(ABC):
    """Abstract interface for authentication."""
    
    @abstractmethod
    async def authenticate(self, token: str) -> Optional[Dict[str, Any]]:
        """Authenticate user token."""
        pass
    
    @abstractmethod
    async def generate_token(self, user_id: str) -> str:
        """Generate JWT token for user."""
        pass
    
    @abstractmethod
    async def validate_api_key(self, api_key: str) -> Optional[str]:
        """Validate API key and return user ID."""
        pass
    
    @abstractmethod
    async def check_rate_limit(self, user_id: str) -> bool:
        """Check if user is within rate limits."""
        pass