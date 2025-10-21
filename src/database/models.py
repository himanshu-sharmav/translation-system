"""
SQLAlchemy models for the machine translation system.
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger, Boolean, CheckConstraint, Column, DateTime, Index, Integer, 
    Numeric, String, Text, UniqueConstraint, text
)
from sqlalchemy.dialects.postgresql import INET, JSONB, UUID as PG_UUID
from sqlalchemy.sql import func

from src.database.connection import Base


class TranslationJob(Base):
    """Model for translation jobs."""
    
    __tablename__ = "translation_jobs"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    source_language = Column(String(10), nullable=False)
    target_language = Column(String(10), nullable=False)
    content_hash = Column(String(64), nullable=False, index=True)
    word_count = Column(Integer, nullable=False)
    priority = Column(String(20), nullable=False, default='normal', index=True)
    status = Column(String(20), nullable=False, default='queued', index=True)
    progress = Column(Numeric(5, 2), nullable=False, default=0.00)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), index=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    estimated_completion = Column(DateTime(timezone=True), nullable=True)
    callback_url = Column(String(500), nullable=True)
    result_url = Column(String(500), nullable=True)
    error_message = Column(Text, nullable=True)
    compute_instance_id = Column(String(100), nullable=True, index=True)
    processing_time_ms = Column(Integer, nullable=True)
    
    # Constraints
    __table_args__ = (
        CheckConstraint('word_count > 0', name='check_word_count_positive'),
        CheckConstraint("priority IN ('normal', 'high', 'critical')", name='check_priority_valid'),
        CheckConstraint("status IN ('queued', 'processing', 'completed', 'failed')", name='check_status_valid'),
        CheckConstraint('progress >= 0.00 AND progress <= 100.00', name='check_progress_range'),
        CheckConstraint('processing_time_ms >= 0', name='check_processing_time_positive'),
        CheckConstraint(
            "(status = 'completed' AND completed_at IS NOT NULL) OR (status != 'completed')",
            name='check_completion_time_consistency'
        ),
        CheckConstraint(
            'started_at IS NULL OR started_at >= created_at',
            name='check_start_time_after_creation'
        ),
        Index('idx_translation_jobs_status_priority', 'status', 'priority'),
        Index('idx_translation_jobs_user_created', 'user_id', 'created_at'),
    )


class TranslationCache(Base):
    """Model for cached translations."""
    
    __tablename__ = "translation_cache"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    content_hash = Column(String(64), nullable=False)
    source_language = Column(String(10), nullable=False)
    target_language = Column(String(10), nullable=False)
    source_content = Column(Text, nullable=False)
    translated_content = Column(Text, nullable=False)
    model_version = Column(String(50), nullable=False, index=True)
    confidence_score = Column(Numeric(5, 2), nullable=True)
    access_count = Column(Integer, nullable=False, default=1, index=True)
    last_accessed = Column(DateTime(timezone=True), nullable=False, default=func.now(), index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), index=True)
    
    # Constraints
    __table_args__ = (
        CheckConstraint('confidence_score >= 0.00 AND confidence_score <= 1.00', name='check_confidence_range'),
        CheckConstraint('access_count >= 0', name='check_access_count_positive'),
        UniqueConstraint(
            'content_hash', 'source_language', 'target_language', 'model_version',
            name='unique_cache_entry'
        ),
        Index('idx_translation_cache_hash_lang', 'content_hash', 'source_language', 'target_language'),
    )


class SystemMetric(Base):
    """Model for system metrics."""
    
    __tablename__ = "system_metrics"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=func.now(), index=True)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Numeric(15, 4), nullable=False)
    instance_id = Column(String(100), nullable=True, index=True)
    tags = Column(JSONB, nullable=True)
    
    # Constraints
    __table_args__ = (
        CheckConstraint("LENGTH(metric_name) > 0", name='check_metric_name_not_empty'),
        Index('idx_system_metrics_timestamp_metric', 'timestamp', 'metric_name'),
        Index('idx_system_metrics_instance_timestamp', 'instance_id', 'timestamp'),
        Index('idx_system_metrics_tags', 'tags', postgresql_using='gin'),
    )


class ComputeInstance(Base):
    """Model for compute instances."""
    
    __tablename__ = "compute_instances"
    
    id = Column(String(100), primary_key=True)
    instance_type = Column(String(50), nullable=False, index=True)
    status = Column(String(20), nullable=False, default='starting', index=True)
    gpu_utilization = Column(Numeric(5, 2), nullable=False, default=0.00)
    memory_usage = Column(Numeric(5, 2), nullable=False, default=0.00)
    cpu_utilization = Column(Numeric(5, 2), nullable=False, default=0.00)
    active_jobs = Column(Integer, nullable=False, default=0, index=True)
    max_concurrent_jobs = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    last_heartbeat = Column(DateTime(timezone=True), nullable=False, default=func.now(), index=True)
    terminated_at = Column(DateTime(timezone=True), nullable=True)
    
    # Constraints
    __table_args__ = (
        CheckConstraint("status IN ('starting', 'running', 'stopping', 'stopped', 'failed')", name='check_status_valid'),
        CheckConstraint('gpu_utilization >= 0.00 AND gpu_utilization <= 100.00', name='check_gpu_utilization_range'),
        CheckConstraint('memory_usage >= 0.00 AND memory_usage <= 100.00', name='check_memory_usage_range'),
        CheckConstraint('cpu_utilization >= 0.00 AND cpu_utilization <= 100.00', name='check_cpu_utilization_range'),
        CheckConstraint('active_jobs >= 0', name='check_active_jobs_positive'),
        CheckConstraint('max_concurrent_jobs > 0', name='check_max_jobs_positive'),
        CheckConstraint(
            "(status = 'stopped' AND terminated_at IS NOT NULL) OR (status != 'stopped')",
            name='check_termination_time_consistency'
        ),
    )


class User(Base):
    """Model for user accounts."""
    
    __tablename__ = "users"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(String(255), nullable=False, unique=True, index=True)
    email = Column(String(255), nullable=True, unique=True, index=True)
    api_key = Column(String(64), nullable=False, unique=True, index=True)
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    rate_limit_per_minute = Column(Integer, nullable=False, default=100)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)
    
    # Constraints
    __table_args__ = (
        CheckConstraint("LENGTH(user_id) > 0", name='check_user_id_not_empty'),
        CheckConstraint("LENGTH(api_key) >= 32", name='check_api_key_length'),
        CheckConstraint('rate_limit_per_minute > 0', name='check_rate_limit_positive'),
    )


class RateLimit(Base):
    """Model for API rate limiting."""
    
    __tablename__ = "rate_limits"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False, index=True)
    endpoint = Column(String(100), nullable=False)
    request_count = Column(Integer, nullable=False, default=1)
    window_start = Column(DateTime(timezone=True), nullable=False, default=func.now())
    window_end = Column(DateTime(timezone=True), nullable=False, 
                       default=text("NOW() + INTERVAL '1 minute'"), index=True)
    
    # Constraints
    __table_args__ = (
        CheckConstraint('request_count > 0', name='check_request_count_positive'),
        UniqueConstraint('user_id', 'endpoint', 'window_start', name='unique_rate_limit_window'),
        Index('idx_rate_limits_user_window', 'user_id', 'window_end'),
    )


class AuditLog(Base):
    """Model for audit logging."""
    
    __tablename__ = "audit_logs"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=func.now(), index=True)
    user_id = Column(String(255), nullable=True, index=True)
    action = Column(String(100), nullable=False, index=True)
    resource_type = Column(String(50), nullable=True)
    resource_id = Column(String(255), nullable=True)
    ip_address = Column(INET, nullable=True)
    user_agent = Column(Text, nullable=True)
    request_data = Column(JSONB, nullable=True)
    response_status = Column(Integer, nullable=True)
    
    # Constraints
    __table_args__ = (
        CheckConstraint("LENGTH(action) > 0", name='check_action_not_empty'),
        Index('idx_audit_logs_user_timestamp', 'user_id', 'timestamp'),
        Index('idx_audit_logs_resource', 'resource_type', 'resource_id'),
        Index('idx_audit_logs_request_data', 'request_data', postgresql_using='gin'),
    )


class QueueMetric(Base):
    """Model for queue performance metrics."""
    
    __tablename__ = "queue_metrics"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=func.now(), index=True)
    priority = Column(String(20), nullable=False, index=True)
    queue_depth = Column(Integer, nullable=False)
    avg_wait_time_seconds = Column(Numeric(10, 2), nullable=True)
    processing_rate_per_minute = Column(Numeric(10, 2), nullable=True)
    
    # Constraints
    __table_args__ = (
        CheckConstraint("priority IN ('normal', 'high', 'critical')", name='check_priority_valid'),
        CheckConstraint('queue_depth >= 0', name='check_queue_depth_positive'),
        CheckConstraint('avg_wait_time_seconds >= 0', name='check_wait_time_positive'),
        CheckConstraint('processing_rate_per_minute >= 0', name='check_processing_rate_positive'),
        UniqueConstraint('timestamp', 'priority', name='unique_queue_metric'),
        Index('idx_queue_metrics_priority_timestamp', 'priority', 'timestamp'),
    )


class CostTracking(Base):
    """Model for cost tracking."""
    
    __tablename__ = "cost_tracking"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=func.now(), index=True)
    date = Column(DateTime(timezone=False), nullable=False, default=func.current_date(), index=True)
    compute_cost_usd = Column(Numeric(10, 4), nullable=False, default=0.0000)
    storage_cost_usd = Column(Numeric(10, 4), nullable=False, default=0.0000)
    network_cost_usd = Column(Numeric(10, 4), nullable=False, default=0.0000)
    words_processed = Column(Integer, nullable=False, default=0)
    jobs_completed = Column(Integer, nullable=False, default=0)
    instance_hours = Column(Numeric(10, 2), nullable=False, default=0.00)
    
    # Constraints
    __table_args__ = (
        CheckConstraint('compute_cost_usd >= 0', name='check_compute_cost_positive'),
        CheckConstraint('storage_cost_usd >= 0', name='check_storage_cost_positive'),
        CheckConstraint('network_cost_usd >= 0', name='check_network_cost_positive'),
        CheckConstraint('words_processed >= 0', name='check_words_processed_positive'),
        CheckConstraint('jobs_completed >= 0', name='check_jobs_completed_positive'),
        CheckConstraint('instance_hours >= 0', name='check_instance_hours_positive'),
        UniqueConstraint('date', name='unique_daily_cost'),
    )
    
    @property
    def total_cost_usd(self) -> Decimal:
        """Calculate total cost."""
        return self.compute_cost_usd + self.storage_cost_usd + self.network_cost_usd