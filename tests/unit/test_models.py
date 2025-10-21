"""
Unit tests for database models.
"""

import pytest
from datetime import datetime
from decimal import Decimal
from uuid import uuid4

from src.database.models import (
    TranslationJob, TranslationCache, SystemMetric, ComputeInstance,
    User, RateLimit, AuditLog, QueueMetric, CostTracking
)


class TestTranslationJob:
    """Test TranslationJob model."""
    
    def test_create_translation_job(self):
        """Test creating a translation job."""
        job_id = uuid4()
        job = TranslationJob(
            id=job_id,
            user_id="test-user",
            source_language="en",
            target_language="es",
            content_hash="test-hash",
            word_count=100,
            priority="normal",
            status="queued"
        )
        
        assert job.id == job_id
        assert job.user_id == "test-user"
        assert job.source_language == "en"
        assert job.target_language == "es"
        assert job.content_hash == "test-hash"
        assert job.word_count == 100
        assert job.priority == "normal"
        assert job.status == "queued"
        assert job.progress == Decimal('0.00')
    
    def test_translation_job_defaults(self):
        """Test translation job default values."""
        job = TranslationJob(
            user_id="test-user",
            source_language="en",
            target_language="es",
            content_hash="test-hash",
            word_count=100
        )
        
        assert job.priority == "normal"
        assert job.status == "queued"
        assert job.progress == Decimal('0.00')
        assert job.created_at is not None
        assert job.started_at is None
        assert job.completed_at is None
    
    def test_translation_job_validation(self):
        """Test translation job field validation."""
        # Test valid priority values
        for priority in ["normal", "high", "critical"]:
            job = TranslationJob(
                user_id="test-user",
                source_language="en",
                target_language="es",
                content_hash="test-hash",
                word_count=100,
                priority=priority
            )
            assert job.priority == priority
        
        # Test valid status values
        for status in ["queued", "processing", "completed", "failed"]:
            job = TranslationJob(
                user_id="test-user",
                source_language="en",
                target_language="es",
                content_hash="test-hash",
                word_count=100,
                status=status
            )
            assert job.status == status


class TestTranslationCache:
    """Test TranslationCache model."""
    
    def test_create_cache_entry(self):
        """Test creating a cache entry."""
        cache_id = uuid4()
        cache = TranslationCache(
            id=cache_id,
            content_hash="test-hash",
            source_language="en",
            target_language="es",
            source_content="Hello world",
            translated_content="Hola mundo",
            model_version="v1.0.0",
            confidence_score=Decimal('0.95')
        )
        
        assert cache.id == cache_id
        assert cache.content_hash == "test-hash"
        assert cache.source_language == "en"
        assert cache.target_language == "es"
        assert cache.source_content == "Hello world"
        assert cache.translated_content == "Hola mundo"
        assert cache.model_version == "v1.0.0"
        assert cache.confidence_score == Decimal('0.95')
    
    def test_cache_entry_defaults(self):
        """Test cache entry default values."""
        cache = TranslationCache(
            content_hash="test-hash",
            source_language="en",
            target_language="es",
            source_content="Hello world",
            translated_content="Hola mundo",
            model_version="v1.0.0"
        )
        
        assert cache.access_count == 1
        assert cache.created_at is not None
        assert cache.last_accessed is not None


class TestSystemMetric:
    """Test SystemMetric model."""
    
    def test_create_system_metric(self):
        """Test creating a system metric."""
        metric = SystemMetric(
            metric_name="cpu_usage",
            metric_value=Decimal('75.5'),
            instance_id="instance-1",
            tags={"environment": "test"}
        )
        
        assert metric.metric_name == "cpu_usage"
        assert metric.metric_value == Decimal('75.5')
        assert metric.instance_id == "instance-1"
        assert metric.tags == {"environment": "test"}
        assert metric.timestamp is not None
    
    def test_system_metric_without_instance(self):
        """Test creating a system metric without instance ID."""
        metric = SystemMetric(
            metric_name="global_metric",
            metric_value=Decimal('100.0')
        )
        
        assert metric.metric_name == "global_metric"
        assert metric.metric_value == Decimal('100.0')
        assert metric.instance_id is None
        assert metric.tags is None


class TestComputeInstance:
    """Test ComputeInstance model."""
    
    def test_create_compute_instance(self):
        """Test creating a compute instance."""
        instance = ComputeInstance(
            id="instance-1",
            instance_type="g4dn.xlarge",
            status="running",
            gpu_utilization=Decimal('80.5'),
            memory_usage=Decimal('65.2'),
            cpu_utilization=Decimal('45.0'),
            active_jobs=3,
            max_concurrent_jobs=5
        )
        
        assert instance.id == "instance-1"
        assert instance.instance_type == "g4dn.xlarge"
        assert instance.status == "running"
        assert instance.gpu_utilization == Decimal('80.5')
        assert instance.memory_usage == Decimal('65.2')
        assert instance.cpu_utilization == Decimal('45.0')
        assert instance.active_jobs == 3
        assert instance.max_concurrent_jobs == 5
    
    def test_compute_instance_defaults(self):
        """Test compute instance default values."""
        instance = ComputeInstance(
            id="instance-1",
            instance_type="g4dn.xlarge"
        )
        
        assert instance.status == "starting"
        assert instance.gpu_utilization == Decimal('0.00')
        assert instance.memory_usage == Decimal('0.00')
        assert instance.cpu_utilization == Decimal('0.00')
        assert instance.active_jobs == 0
        assert instance.max_concurrent_jobs == 1
        assert instance.created_at is not None
        assert instance.last_heartbeat is not None


class TestUser:
    """Test User model."""
    
    def test_create_user(self):
        """Test creating a user."""
        user_id = uuid4()
        user = User(
            id=user_id,
            user_id="test-user",
            email="test@example.com",
            api_key="test-api-key-12345678901234567890123456789012",
            is_active=True,
            rate_limit_per_minute=200
        )
        
        assert user.id == user_id
        assert user.user_id == "test-user"
        assert user.email == "test@example.com"
        assert user.api_key == "test-api-key-12345678901234567890123456789012"
        assert user.is_active is True
        assert user.rate_limit_per_minute == 200
    
    def test_user_defaults(self):
        """Test user default values."""
        user = User(
            user_id="test-user",
            api_key="test-api-key-12345678901234567890123456789012"
        )
        
        assert user.is_active is True
        assert user.rate_limit_per_minute == 100
        assert user.created_at is not None
        assert user.last_login is None


class TestRateLimit:
    """Test RateLimit model."""
    
    def test_create_rate_limit(self):
        """Test creating a rate limit entry."""
        rate_limit = RateLimit(
            user_id="test-user",
            endpoint="/api/v1/translate",
            request_count=5
        )
        
        assert rate_limit.user_id == "test-user"
        assert rate_limit.endpoint == "/api/v1/translate"
        assert rate_limit.request_count == 5
        assert rate_limit.window_start is not None
        assert rate_limit.window_end is not None
    
    def test_rate_limit_defaults(self):
        """Test rate limit default values."""
        rate_limit = RateLimit(
            user_id="test-user",
            endpoint="/api/v1/translate"
        )
        
        assert rate_limit.request_count == 1


class TestAuditLog:
    """Test AuditLog model."""
    
    def test_create_audit_log(self):
        """Test creating an audit log entry."""
        audit_log = AuditLog(
            user_id="test-user",
            action="create_translation_job",
            resource_type="translation_job",
            resource_id="job-123",
            ip_address="192.168.1.1",
            user_agent="Test Agent",
            request_data={"source_language": "en", "target_language": "es"},
            response_status=201
        )
        
        assert audit_log.user_id == "test-user"
        assert audit_log.action == "create_translation_job"
        assert audit_log.resource_type == "translation_job"
        assert audit_log.resource_id == "job-123"
        assert audit_log.ip_address == "192.168.1.1"
        assert audit_log.user_agent == "Test Agent"
        assert audit_log.request_data == {"source_language": "en", "target_language": "es"}
        assert audit_log.response_status == 201
        assert audit_log.timestamp is not None


class TestQueueMetric:
    """Test QueueMetric model."""
    
    def test_create_queue_metric(self):
        """Test creating a queue metric."""
        queue_metric = QueueMetric(
            priority="high",
            queue_depth=15,
            avg_wait_time_seconds=Decimal('45.5'),
            processing_rate_per_minute=Decimal('12.3')
        )
        
        assert queue_metric.priority == "high"
        assert queue_metric.queue_depth == 15
        assert queue_metric.avg_wait_time_seconds == Decimal('45.5')
        assert queue_metric.processing_rate_per_minute == Decimal('12.3')
        assert queue_metric.timestamp is not None


class TestCostTracking:
    """Test CostTracking model."""
    
    def test_create_cost_tracking(self):
        """Test creating a cost tracking entry."""
        cost_tracking = CostTracking(
            compute_cost_usd=Decimal('25.50'),
            storage_cost_usd=Decimal('5.25'),
            network_cost_usd=Decimal('2.10'),
            words_processed=50000,
            jobs_completed=125,
            instance_hours=Decimal('8.5')
        )
        
        assert cost_tracking.compute_cost_usd == Decimal('25.50')
        assert cost_tracking.storage_cost_usd == Decimal('5.25')
        assert cost_tracking.network_cost_usd == Decimal('2.10')
        assert cost_tracking.words_processed == 50000
        assert cost_tracking.jobs_completed == 125
        assert cost_tracking.instance_hours == Decimal('8.5')
        assert cost_tracking.timestamp is not None
        assert cost_tracking.date is not None
    
    def test_cost_tracking_total_cost(self):
        """Test cost tracking total cost calculation."""
        cost_tracking = CostTracking(
            compute_cost_usd=Decimal('25.50'),
            storage_cost_usd=Decimal('5.25'),
            network_cost_usd=Decimal('2.10')
        )
        
        expected_total = Decimal('25.50') + Decimal('5.25') + Decimal('2.10')
        assert cost_tracking.total_cost_usd == expected_total
    
    def test_cost_tracking_defaults(self):
        """Test cost tracking default values."""
        cost_tracking = CostTracking()
        
        assert cost_tracking.compute_cost_usd == Decimal('0.0000')
        assert cost_tracking.storage_cost_usd == Decimal('0.0000')
        assert cost_tracking.network_cost_usd == Decimal('0.0000')
        assert cost_tracking.words_processed == 0
        assert cost_tracking.jobs_completed == 0
        assert cost_tracking.instance_hours == Decimal('0.00')
        assert cost_tracking.total_cost_usd == Decimal('0.0000')