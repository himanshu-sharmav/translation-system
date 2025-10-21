"""
Unit tests for JobRepository.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from src.config.config import JobStatus, Priority
from src.database.models import TranslationJob
from src.utils.exceptions import DatabaseError
from tests.conftest import assert_job_equals


class TestJobRepository:
    """Test JobRepository operations."""
    
    @pytest.mark.asyncio
    async def test_create_job(self, job_repository, sample_translation_job):
        """Test creating a translation job."""
        created_job = await job_repository.create(sample_translation_job)
        
        assert created_job.id == sample_translation_job.id
        assert created_job.user_id == sample_translation_job.user_id
        assert created_job.status == "queued"
        assert created_job.created_at is not None
    
    @pytest.mark.asyncio
    async def test_get_job_by_id(self, job_repository, sample_translation_job):
        """Test getting a job by ID."""
        # Create job first
        await job_repository.create(sample_translation_job)
        
        # Retrieve job
        retrieved_job = await job_repository.get_by_id(sample_translation_job.id)
        
        assert retrieved_job is not None
        assert_job_equals(retrieved_job, sample_translation_job)
    
    @pytest.mark.asyncio
    async def test_get_job_by_id_not_found(self, job_repository):
        """Test getting a non-existent job."""
        non_existent_id = uuid4()
        job = await job_repository.get_by_id(non_existent_id)
        assert job is None
    
    @pytest.mark.asyncio
    async def test_update_job(self, job_repository, sample_translation_job):
        """Test updating a job."""
        # Create job first
        await job_repository.create(sample_translation_job)
        
        # Update job
        sample_translation_job.status = "processing"
        sample_translation_job.progress = 50.0
        
        success = await job_repository.update(sample_translation_job)
        assert success is True
        
        # Verify update
        updated_job = await job_repository.get_by_id(sample_translation_job.id)
        assert updated_job.status == "processing"
        assert updated_job.progress == 50.0
    
    @pytest.mark.asyncio
    async def test_delete_job(self, job_repository, sample_translation_job):
        """Test deleting a job."""
        # Create job first
        await job_repository.create(sample_translation_job)
        
        # Delete job
        success = await job_repository.delete(sample_translation_job.id)
        assert success is True
        
        # Verify deletion
        deleted_job = await job_repository.get_by_id(sample_translation_job.id)
        assert deleted_job is None
    
    @pytest.mark.asyncio
    async def test_get_by_status(self, job_repository, db_session):
        """Test getting jobs by status."""
        # Create jobs with different statuses
        jobs = []
        for i, status in enumerate(["queued", "processing", "completed"]):
            job = TranslationJob(
                id=uuid4(),
                user_id=f"user-{i}",
                source_language="en",
                target_language="es",
                content_hash=f"hash-{i}",
                word_count=100,
                status=status
            )
            jobs.append(job)
            await job_repository.create(job)
        
        # Test getting queued jobs
        queued_jobs = await job_repository.get_by_status(JobStatus.QUEUED)
        assert len(queued_jobs) == 1
        assert queued_jobs[0].status == "queued"
        
        # Test getting processing jobs
        processing_jobs = await job_repository.get_by_status(JobStatus.PROCESSING)
        assert len(processing_jobs) == 1
        assert processing_jobs[0].status == "processing"
    
    @pytest.mark.asyncio
    async def test_get_by_user(self, job_repository):
        """Test getting jobs by user."""
        user_id = "test-user"
        
        # Create jobs for the user
        jobs = []
        for i in range(3):
            job = TranslationJob(
                id=uuid4(),
                user_id=user_id,
                source_language="en",
                target_language="es",
                content_hash=f"hash-{i}",
                word_count=100
            )
            jobs.append(job)
            await job_repository.create(job)
        
        # Create job for different user
        other_job = TranslationJob(
            id=uuid4(),
            user_id="other-user",
            source_language="en",
            target_language="es",
            content_hash="other-hash",
            word_count=100
        )
        await job_repository.create(other_job)
        
        # Get jobs for the user
        user_jobs = await job_repository.get_by_user(user_id)
        assert len(user_jobs) == 3
        
        for job in user_jobs:
            assert job.user_id == user_id
    
    @pytest.mark.asyncio
    async def test_get_queue_depth_by_priority(self, job_repository):
        """Test getting queue depth by priority."""
        # Create jobs with different priorities
        priorities = [Priority.NORMAL, Priority.HIGH, Priority.CRITICAL]
        
        for priority in priorities:
            for i in range(2):  # 2 jobs per priority
                job = TranslationJob(
                    id=uuid4(),
                    user_id=f"user-{priority.value}-{i}",
                    source_language="en",
                    target_language="es",
                    content_hash=f"hash-{priority.value}-{i}",
                    word_count=100,
                    priority=priority.value,
                    status="queued"
                )
                await job_repository.create(job)
        
        # Test queue depths
        normal_depth = await job_repository.get_queue_depth_by_priority(Priority.NORMAL)
        high_depth = await job_repository.get_queue_depth_by_priority(Priority.HIGH)
        critical_depth = await job_repository.get_queue_depth_by_priority(Priority.CRITICAL)
        
        assert normal_depth == 2
        assert high_depth == 2
        assert critical_depth == 2
    
    @pytest.mark.asyncio
    async def test_get_next_job_by_priority(self, job_repository):
        """Test getting next job by priority order."""
        # Create jobs with different priorities and timestamps
        jobs = []
        
        # Normal priority job (created first)
        normal_job = TranslationJob(
            id=uuid4(),
            user_id="user-normal",
            source_language="en",
            target_language="es",
            content_hash="hash-normal",
            word_count=100,
            priority="normal",
            status="queued"
        )
        jobs.append(normal_job)
        await job_repository.create(normal_job)
        
        # High priority job (created second)
        high_job = TranslationJob(
            id=uuid4(),
            user_id="user-high",
            source_language="en",
            target_language="es",
            content_hash="hash-high",
            word_count=100,
            priority="high",
            status="queued"
        )
        jobs.append(high_job)
        await job_repository.create(high_job)
        
        # Critical priority job (created last)
        critical_job = TranslationJob(
            id=uuid4(),
            user_id="user-critical",
            source_language="en",
            target_language="es",
            content_hash="hash-critical",
            word_count=100,
            priority="critical",
            status="queued"
        )
        jobs.append(critical_job)
        await job_repository.create(critical_job)
        
        # Get next job - should be critical priority despite being created last
        next_job = await job_repository.get_next_job_by_priority()
        assert next_job is not None
        assert next_job.priority == "critical"
        assert next_job.id == critical_job.id
    
    @pytest.mark.asyncio
    async def test_update_job_status(self, job_repository, sample_translation_job):
        """Test updating job status with related fields."""
        # Create job
        await job_repository.create(sample_translation_job)
        
        # Update to processing
        success = await job_repository.update_job_status(
            sample_translation_job.id,
            JobStatus.PROCESSING,
            progress=25.0,
            compute_instance_id="instance-1"
        )
        assert success is True
        
        # Verify update
        updated_job = await job_repository.get_by_id(sample_translation_job.id)
        assert updated_job.status == "processing"
        assert updated_job.progress == 25.0
        assert updated_job.compute_instance_id == "instance-1"
        assert updated_job.started_at is not None
        
        # Update to completed
        success = await job_repository.update_job_status(
            sample_translation_job.id,
            JobStatus.COMPLETED,
            processing_time_ms=5000
        )
        assert success is True
        
        # Verify completion
        completed_job = await job_repository.get_by_id(sample_translation_job.id)
        assert completed_job.status == "completed"
        assert completed_job.progress == 100.0
        assert completed_job.processing_time_ms == 5000
        assert completed_job.completed_at is not None
    
    @pytest.mark.asyncio
    async def test_update_job_status_not_found(self, job_repository):
        """Test updating status of non-existent job."""
        non_existent_id = uuid4()
        success = await job_repository.update_job_status(
            non_existent_id,
            JobStatus.PROCESSING
        )
        assert success is False
    
    @pytest.mark.asyncio
    async def test_get_processing_jobs(self, job_repository):
        """Test getting currently processing jobs."""
        # Create jobs with different statuses
        processing_job1 = TranslationJob(
            id=uuid4(),
            user_id="user-1",
            source_language="en",
            target_language="es",
            content_hash="hash-1",
            word_count=100,
            status="processing",
            compute_instance_id="instance-1"
        )
        
        processing_job2 = TranslationJob(
            id=uuid4(),
            user_id="user-2",
            source_language="en",
            target_language="es",
            content_hash="hash-2",
            word_count=100,
            status="processing",
            compute_instance_id="instance-2"
        )
        
        queued_job = TranslationJob(
            id=uuid4(),
            user_id="user-3",
            source_language="en",
            target_language="es",
            content_hash="hash-3",
            word_count=100,
            status="queued"
        )
        
        await job_repository.create(processing_job1)
        await job_repository.create(processing_job2)
        await job_repository.create(queued_job)
        
        # Get all processing jobs
        processing_jobs = await job_repository.get_processing_jobs()
        assert len(processing_jobs) == 2
        
        # Get processing jobs for specific instance
        instance_jobs = await job_repository.get_processing_jobs("instance-1")
        assert len(instance_jobs) == 1
        assert instance_jobs[0].compute_instance_id == "instance-1"
    
    @pytest.mark.asyncio
    async def test_get_stale_jobs(self, job_repository):
        """Test getting stale processing jobs."""
        # Create a job that started processing long ago
        old_time = datetime.utcnow() - timedelta(hours=2)
        stale_job = TranslationJob(
            id=uuid4(),
            user_id="user-stale",
            source_language="en",
            target_language="es",
            content_hash="hash-stale",
            word_count=100,
            status="processing",
            started_at=old_time
        )
        
        # Create a recent processing job
        recent_job = TranslationJob(
            id=uuid4(),
            user_id="user-recent",
            source_language="en",
            target_language="es",
            content_hash="hash-recent",
            word_count=100,
            status="processing",
            started_at=datetime.utcnow()
        )
        
        await job_repository.create(stale_job)
        await job_repository.create(recent_job)
        
        # Get stale jobs (timeout 30 minutes)
        stale_jobs = await job_repository.get_stale_jobs(30)
        assert len(stale_jobs) == 1
        assert stale_jobs[0].id == stale_job.id
    
    @pytest.mark.asyncio
    async def test_get_job_statistics(self, job_repository):
        """Test getting job statistics."""
        user_id = "test-user"
        
        # Create jobs with different statuses
        completed_job = TranslationJob(
            id=uuid4(),
            user_id=user_id,
            source_language="en",
            target_language="es",
            content_hash="hash-completed",
            word_count=100,
            status="completed",
            processing_time_ms=5000
        )
        
        failed_job = TranslationJob(
            id=uuid4(),
            user_id=user_id,
            source_language="en",
            target_language="es",
            content_hash="hash-failed",
            word_count=200,
            status="failed"
        )
        
        queued_job = TranslationJob(
            id=uuid4(),
            user_id=user_id,
            source_language="en",
            target_language="es",
            content_hash="hash-queued",
            word_count=150,
            status="queued"
        )
        
        await job_repository.create(completed_job)
        await job_repository.create(failed_job)
        await job_repository.create(queued_job)
        
        # Get statistics for user
        stats = await job_repository.get_job_statistics(user_id, days=30)
        
        assert stats["completed"] == 1
        assert stats["failed"] == 1
        assert stats["queued"] == 1
        assert stats["total_words"] == 450  # 100 + 200 + 150
        assert stats["avg_processing_time_ms"] == 5000.0
    
    @pytest.mark.asyncio
    async def test_get_queue_position(self, job_repository):
        """Test getting queue position for a job."""
        # Create jobs with different priorities
        jobs = []
        
        # Critical priority job
        critical_job = TranslationJob(
            id=uuid4(),
            user_id="user-critical",
            source_language="en",
            target_language="es",
            content_hash="hash-critical",
            word_count=100,
            priority="critical",
            status="queued"
        )
        jobs.append(critical_job)
        await job_repository.create(critical_job)
        
        # High priority job
        high_job = TranslationJob(
            id=uuid4(),
            user_id="user-high",
            source_language="en",
            target_language="es",
            content_hash="hash-high",
            word_count=100,
            priority="high",
            status="queued"
        )
        jobs.append(high_job)
        await job_repository.create(high_job)
        
        # Normal priority job
        normal_job = TranslationJob(
            id=uuid4(),
            user_id="user-normal",
            source_language="en",
            target_language="es",
            content_hash="hash-normal",
            word_count=100,
            priority="normal",
            status="queued"
        )
        jobs.append(normal_job)
        await job_repository.create(normal_job)
        
        # Check queue positions
        critical_position = await job_repository.get_queue_position(critical_job.id)
        high_position = await job_repository.get_queue_position(high_job.id)
        normal_position = await job_repository.get_queue_position(normal_job.id)
        
        assert critical_position == 1  # First in queue
        assert high_position == 2      # Second in queue
        assert normal_position == 3    # Third in queue
    
    @pytest.mark.asyncio
    async def test_cleanup_old_jobs(self, job_repository):
        """Test cleaning up old completed jobs."""
        # Create old completed job
        old_time = datetime.utcnow() - timedelta(days=35)
        old_job = TranslationJob(
            id=uuid4(),
            user_id="user-old",
            source_language="en",
            target_language="es",
            content_hash="hash-old",
            word_count=100,
            status="completed",
            completed_at=old_time
        )
        
        # Create recent completed job
        recent_job = TranslationJob(
            id=uuid4(),
            user_id="user-recent",
            source_language="en",
            target_language="es",
            content_hash="hash-recent",
            word_count=100,
            status="completed",
            completed_at=datetime.utcnow()
        )
        
        await job_repository.create(old_job)
        await job_repository.create(recent_job)
        
        # Cleanup jobs older than 30 days
        deleted_count = await job_repository.cleanup_old_jobs(days=30)
        assert deleted_count == 1
        
        # Verify old job is deleted, recent job remains
        old_job_check = await job_repository.get_by_id(old_job.id)
        recent_job_check = await job_repository.get_by_id(recent_job.id)
        
        assert old_job_check is None
        assert recent_job_check is not None
    
    @pytest.mark.asyncio
    async def test_get_performance_metrics(self, job_repository):
        """Test getting performance metrics."""
        # Create completed jobs
        for i in range(3):
            job = TranslationJob(
                id=uuid4(),
                user_id=f"user-{i}",
                source_language="en",
                target_language="es",
                content_hash=f"hash-{i}",
                word_count=100,
                status="completed",
                processing_time_ms=5000,
                completed_at=datetime.utcnow()
            )
            await job_repository.create(job)
        
        # Get performance metrics
        metrics = await job_repository.get_performance_metrics(hours=24)
        
        assert metrics["total_jobs"] == 3
        assert metrics["avg_processing_time_ms"] == 5000.0
        assert metrics["total_words_processed"] == 300
        assert metrics["avg_words_per_minute"] > 0
        assert metrics["throughput_jobs_per_hour"] > 0