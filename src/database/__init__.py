"""
Database package for the machine translation system.
"""

from .connection import (
    DatabaseManager,
    db_manager,
    init_database,
    close_database,
    get_db_session,
    get_db_connection,
    TransactionManager,
    with_transaction
)

from .models import (
    Base,
    TranslationJob,
    TranslationCache,
    SystemMetric,
    ComputeInstance,
    User,
    RateLimit,
    AuditLog,
    QueueMetric,
    CostTracking
)

from .repositories import (
    BaseRepository,
    RepositoryMixin,
    JobRepository,
    CacheRepository,
    MetricsRepository
)

__all__ = [
    # Connection management
    "DatabaseManager",
    "db_manager",
    "init_database",
    "close_database",
    "get_db_session",
    "get_db_connection",
    "TransactionManager",
    "with_transaction",
    
    # Models
    "Base",
    "TranslationJob",
    "TranslationCache",
    "SystemMetric",
    "ComputeInstance",
    "User",
    "RateLimit",
    "AuditLog",
    "QueueMetric",
    "CostTracking",
    
    # Repositories
    "BaseRepository",
    "RepositoryMixin",
    "JobRepository",
    "CacheRepository",
    "MetricsRepository"
]