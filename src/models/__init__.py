"""
Models package for the machine translation backend system.
"""

from .interfaces import (
    TranslationRequest,
    TranslationJob,
    TranslationResult,
    CacheEntry,
    SystemMetric,
    ComputeInstance,
    TranslationEngine,
    QueueManager,
    CacheManager,
    ResourceManager,
    Repository,
    JobRepository,
    CacheRepository,
    MetricsRepository,
    MonitoringService,
    AuthenticationService
)

__all__ = [
    "TranslationRequest",
    "TranslationJob", 
    "TranslationResult",
    "CacheEntry",
    "SystemMetric",
    "ComputeInstance",
    "TranslationEngine",
    "QueueManager",
    "CacheManager",
    "ResourceManager",
    "Repository",
    "JobRepository",
    "CacheRepository",
    "MetricsRepository",
    "MonitoringService",
    "AuthenticationService"
]