"""
Pytest configuration and fixtures for the machine translation system tests.
"""

import asyncio
import os
from typing import AsyncGenerator
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool

from src.config.config import Config, DatabaseConfig, Environment
from src.database.connection import Base, DatabaseManager
from src.database.models import TranslationJob, TranslationCache, SystemMetric, User
from src.database.repositories import JobRepository, CacheRepository, MetricsRepository


# Test database configuration
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def test_engine():
    """Create test database engine."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        echo=False
    )
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Cleanup
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    session_factory = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    async with session_factory() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def job_repository(db_session) -> JobRepository:
    """Create a job repository instance."""
    return JobRepository(db_session)


@pytest_asyncio.fixture
async def cache_repository(db_session) -> CacheRepository:
    """Create a cache repository instance."""
    return CacheRepository(db_session)


@pytest_asyncio.fixture
async def metrics_repository(db_session) -> MetricsRepository:
    """Create a metrics repository instance."""
    return MetricsRepository(db_session)


@pytest.fixture
def sample_translation_job():
    """Create a sample translation job for testing."""
    return TranslationJob(
        id=uuid4(),
        user_id="test-user",
        source_language="en",
        target_language="es",
        content_hash="test-hash-123",
        word_count=100,
        priority="normal",
        status="queued"
    )


@pytest.fixture
def sample_cache_entry():
    """Create a sample cache entry for testing."""
    return TranslationCache(
        id=uuid4(),
        content_hash="test-hash-123",
        source_language="en",
        target_language="es",
        source_content="Hello world",
        translated_content="Hola mundo",
        model_version="v1.0.0",
        confidence_score=0.95
    )


@pytest.fixture
def sample_system_metric():
    """Create a sample system metric for testing."""
    return SystemMetric(
        metric_name="test_metric",
        metric_value=75.5,
        instance_id="test-instance-1",
        tags={"environment": "test"}
    )


@pytest.fixture
def sample_user():
    """Create a sample user for testing."""
    return User(
        id=uuid4(),
        user_id="test-user",
        email="test@example.com",
        api_key="test-api-key-12345678901234567890123456789012",
        is_active=True,
        rate_limit_per_minute=100
    )


@pytest.fixture
def test_config():
    """Create test configuration."""
    return Config(
        environment=Environment.DEVELOPMENT,
        database=DatabaseConfig(
            host="localhost",
            port=5432,
            database="test_db",
            username="test",
            password="test"
        ),
        redis=None,
        compute=None,
        performance=None,
        security=None,
        monitoring=None
    )


# Utility functions for tests
def assert_job_equals(job1: TranslationJob, job2: TranslationJob, ignore_timestamps=True):
    """Assert that two translation jobs are equal."""
    assert job1.id == job2.id
    assert job1.user_id == job2.user_id
    assert job1.source_language == job2.source_language
    assert job1.target_language == job2.target_language
    assert job1.content_hash == job2.content_hash
    assert job1.word_count == job2.word_count
    assert job1.priority == job2.priority
    assert job1.status == job2.status
    assert job1.progress == job2.progress
    
    if not ignore_timestamps:
        assert job1.created_at == job2.created_at
        assert job1.started_at == job2.started_at
        assert job1.completed_at == job2.completed_at


def assert_cache_equals(cache1: TranslationCache, cache2: TranslationCache, ignore_timestamps=True):
    """Assert that two cache entries are equal."""
    assert cache1.id == cache2.id
    assert cache1.content_hash == cache2.content_hash
    assert cache1.source_language == cache2.source_language
    assert cache1.target_language == cache2.target_language
    assert cache1.source_content == cache2.source_content
    assert cache1.translated_content == cache2.translated_content
    assert cache1.model_version == cache2.model_version
    assert cache1.confidence_score == cache2.confidence_score
    assert cache1.access_count == cache2.access_count
    
    if not ignore_timestamps:
        assert cache1.created_at == cache2.created_at
        assert cache1.last_accessed == cache2.last_accessed


def assert_metric_equals(metric1: SystemMetric, metric2: SystemMetric, ignore_timestamps=True):
    """Assert that two system metrics are equal."""
    assert metric1.metric_name == metric2.metric_name
    assert metric1.metric_value == metric2.metric_value
    assert metric1.instance_id == metric2.instance_id
    assert metric1.tags == metric2.tags
    
    if not ignore_timestamps:
        assert metric1.timestamp == metric2.timestamp