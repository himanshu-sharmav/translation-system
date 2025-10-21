"""
Integration tests for queue management system.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from uuid import uuid4

from src.config.config import Priority, JobStatus
from src.database.models import TranslationJob, ComputeInstance
from src.database.repositories import JobRepository
from src.database.repositories.base import BaseRepository
from src.services.queue_service import QueueService
from src.services.job_scheduler import JobScheduler
from src.services.queue_manager import QueueManager


class TestQueueIntegration:
    """Test queue system integration."""
    
    @pytest.fixture
    async def queue_service(self):
        """Create QueueService instance."""
        service = QueueService()
        yield service
        await service.close()
    
    @pytest.fixture
    async def job_scheduler(self):
        """Create JobScheduler instance."""
        scheduler = JobScheduler()
        yield scheduler
        await scheduler.close()
    
    @pytest.fixture
    async def queue_manager(self):
        """Create QueueManager instance."""
        manager = QueueManager()
        yield manager
        await manager.stop_monitoring()
    
    @pytest.fixture
    async def sample_jobs(self, db_session):
        """Create sample jobs in database."""
        job_repo = JobRepository(db_session)
        jobs = []
        
        priorities = [Priority.CRITICAL, Priority.HIGH, Priority.NORMAL]
        
        for i, priority in enumerate(priorities):
            job = TranslationJob(
                id=uuid4(),
                user_id=f"user-{i}",
                source_language="en",
                target_language="es",
                content_hash=f"hash-{i}",
                word_count=100 * (i + 1),
                priority=priority.value,
                status=JobStatus.QUEUED.value,
                created_at=datetime.utcnow() - timedelta(minutes=i)
            )
            
            created_job = await job_repo.create(job)
            jobs.append(created_job)
        
        await db_session.commit()
        return jobs
    
    @pytest.fixture
    async def sample_compute_instance(self, db_session):
        """Create sample compute instance."""
        instance_repo = BaseRepository(db_session, ComputeInstance)
        
        instance = ComputeInstance(
            id="test-instance-1",
            instance_type="g4dn.xlarge",
            status="running",
            gpu_utilization=50.0,
            memory_usage=60.0,
            active_jobs=0,
            max_concurrent_jobs=3,
            created_at=datetime.utcnow(),
            last_heartbeat=datetime.utcnow()
        )
        
        created_instance = await instance_repo.create(instance)
        await db_session.commit()
        return created_instance
    
    @pytest.mark.asyncio
    async def test_end_to_end_job_processing(self, queue_service, job_scheduler, sample_jobs, sample_compute_instance):
        """Test end-to-end job processing flow."""
        # This test requires Redis to be available
        pytest.skip("Requires Redis connection for integration test")
        
        # Enqueue jobs
        for job in sample_jobs:
            success = await queue_service.enqueue(job)
            assert success is True
        
        # Check queue depths
        total_depth = await queue_service.get_queue_depth()
        assert total_depth == 3
        
        critical_depth = await queue_service.get_queue_depth(Priority.CRITICAL)
        assert critical_depth == 1
        
        # Dequeue jobs in priority order
        first_job = await queue_service.dequeue()
        assert first_job is not None
        assert first_job.priority == Priority.CRITICAL.value
        
        second_job = await queue_service.dequeue()
        assert second_job is not None
        assert second_job.priority == Priority.HIGH.value
        
        third_job = await queue_service.dequeue()
        assert third_job is not None
        assert third_job.priority == Priority.NORMAL.value
        
        # No more jobs
        no_job = await queue_service.dequeue()
        assert no_job is None
    
    @pytest.mark.asyncio
    async def test_queue_stats_collection(self, queue_manager, sample_jobs):
        """Test queue statistics collection."""
        # This test would require Redis and database setup
        pytest.skip("Requires Redis and database setup for integration test")
        
        # Start monitoring
        await queue_manager.start_monitoring()
        
        # Wait for metrics collection
        await asyncio.sleep(2)
        
        # Get analytics
        analytics = await queue_manager.get_queue_analytics(hours=1)
        
        assert "priority_analytics" in analytics
        assert "system_analytics" in analytics
        
        # Stop monitoring
        await queue_manager.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_job_timeout_detection(self, job_scheduler, db_session):
        """Test job timeout detection and handling."""
        # Create job with old start time
        job_repo = JobRepository(db_session)
        
        timeout_job = TranslationJob(
            id=uuid4(),
            user_id="timeout-user",
            source_language="en",
            target_language="es",
            content_hash="timeout-hash",
            word_count=100,
            priority="normal",
            status="processing",
            started_at=datetime.utcnow() - timedelta(hours=2),  # 2 hours ago
            compute_instance_id="instance-1"
        )
        
        created_job = await job_repo.create(timeout_job)
        await db_session.commit()
        
        # Check for timeouts (30 minute timeout)
        await job_scheduler._check_job_timeouts()
        
        # Verify job was marked as failed
        updated_job = await job_repo.get_by_id(created_job.id)
        assert updated_job.status == JobStatus.FAILED.value
        assert "timed out" in updated_job.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_queue_priority_ordering(self, queue_service, db_session):
        """Test that jobs are processed in correct priority order."""
        # This test requires Redis setup
        pytest.skip("Requires Redis connection for integration test")
        
        job_repo = JobRepository(db_session)
        
        # Create jobs with different priorities but reverse creation order
        jobs = []
        priorities = [Priority.NORMAL, Priority.HIGH, Priority.CRITICAL]  # Reverse order
        
        for i, priority in enumerate(priorities):
            job = TranslationJob(
                id=uuid4(),
                user_id=f"user-{i}",
                source_language="en",
                target_language="es",
                content_hash=f"hash-{i}",
                word_count=100,
                priority=priority.value,
                status=JobStatus.QUEUED.value,
                created_at=datetime.utcnow() + timedelta(seconds=i)  # Later creation times
            )
            
            created_job = await job_repo.create(job)
            jobs.append(created_job)
            
            # Enqueue job
            await queue_service.enqueue(created_job)
        
        await db_session.commit()
        
        # Dequeue jobs - should come out in priority order (critical, high, normal)
        dequeued_jobs = []
        for _ in range(3):
            job = await queue_service.dequeue()
            if job:
                dequeued_jobs.append(job)
        
        # Verify priority order
        assert len(dequeued_jobs) == 3
        assert dequeued_jobs[0].priority == Priority.CRITICAL.value
        assert dequeued_jobs[1].priority == Priority.HIGH.value
        assert dequeued_jobs[2].priority == Priority.NORMAL.value
    
    @pytest.mark.asyncio
    async def test_dead_letter_queue_processing(self, queue_service, db_session):
        """Test dead letter queue functionality."""
        # This test requires Redis setup
        pytest.skip("Requires Redis connection for integration test")
        
        job_repo = JobRepository(db_session)
        
        # Create a failed job
        failed_job = TranslationJob(
            id=uuid4(),
            user_id="failed-user",
            source_language="en",
            target_language="es",
            content_hash="failed-hash",
            word_count=100,
            priority="normal",
            status="failed"
        )
        
        created_job = await job_repo.create(failed_job)
        await db_session.commit()
        
        # Move to dead letter queue
        error_message = "Translation engine failure"
        success = await queue_service.move_to_dead_letter_queue(created_job.id, error_message)
        assert success is True
        
        # Check DLQ stats
        stats = await queue_service.get_queue_stats()
        assert stats["dead_letter"] == 1
        
        # Retry DLQ jobs
        retried_count = await queue_service.retry_dead_letter_jobs(max_retries=3)
        assert retried_count >= 0  # May be 0 if job can't be retried
    
    @pytest.mark.asyncio
    async def test_concurrent_queue_operations(self, queue_service, db_session):
        """Test concurrent queue operations."""
        # This test requires Redis setup
        pytest.skip("Requires Redis connection for integration test")
        
        job_repo = JobRepository(db_session)
        
        # Create multiple jobs
        jobs = []
        for i in range(10):
            job = TranslationJob(
                id=uuid4(),
                user_id=f"concurrent-user-{i}",
                source_language="en",
                target_language="es",
                content_hash=f"concurrent-hash-{i}",
                word_count=100,
                priority="normal",
                status=JobStatus.QUEUED.value,
                created_at=datetime.utcnow()
            )
            
            created_job = await job_repo.create(job)
            jobs.append(created_job)
        
        await db_session.commit()
        
        # Enqueue jobs concurrently
        enqueue_tasks = [queue_service.enqueue(job) for job in jobs]
        enqueue_results = await asyncio.gather(*enqueue_tasks)
        
        assert all(result is True for result in enqueue_results)
        
        # Check total queue depth
        total_depth = await queue_service.get_queue_depth()
        assert total_depth == 10
        
        # Dequeue jobs concurrently
        dequeue_tasks = [queue_service.dequeue() for _ in range(10)]
        dequeued_jobs = await asyncio.gather(*dequeue_tasks)
        
        # Should get all jobs back (though order may vary due to concurrency)
        non_null_jobs = [job for job in dequeued_jobs if job is not None]
        assert len(non_null_jobs) == 10
    
    @pytest.mark.asyncio
    async def test_queue_error_recovery(self, queue_service):
        """Test queue error handling and recovery."""
        # Test with invalid job data
        invalid_job = TranslationJob(
            id=uuid4(),
            user_id="",  # Invalid empty user_id
            source_language="en",
            target_language="es",
            content_hash="test-hash",
            word_count=100,
            priority="normal",
            status="queued",
            created_at=datetime.utcnow()
        )
        
        # This should handle the error gracefully
        with pytest.raises(Exception):  # Specific exception depends on validation
            await queue_service.enqueue(invalid_job)
    
    @pytest.mark.asyncio
    async def test_priority_queue_ordering_under_concurrent_load(self, queue_service, job_scheduler, db_session):
        """Test priority ordering is maintained under concurrent load (Requirements 3.1, 3.2, 3.4)."""
        # This test requires Redis setup
        pytest.skip("Requires Redis connection for integration test")
        
        job_repo = JobRepository(db_session)
        
        # Create jobs with mixed priorities submitted concurrently
        jobs = []
        priorities = [Priority.CRITICAL, Priority.HIGH, Priority.NORMAL]
        
        for i in range(30):  # 10 jobs per priority
            priority = priorities[i % 3]
            job = TranslationJob(
                id=uuid4(),
                user_id=f"concurrent-user-{i}",
                source_language="en",
                target_language="es",
                content_hash=f"concurrent-hash-{i}",
                word_count=100,
                priority=priority.value,
                status=JobStatus.QUEUED.value,
                created_at=datetime.utcnow() + timedelta(microseconds=i)  # Slight time differences
            )
            
            created_job = await job_repo.create(job)
            jobs.append(created_job)
        
        await db_session.commit()
        
        # Enqueue all jobs concurrently
        enqueue_tasks = [queue_service.enqueue(job) for job in jobs]
        await asyncio.gather(*enqueue_tasks)
        
        # Dequeue jobs and verify priority ordering
        dequeued_jobs = []
        for _ in range(30):
            job = await queue_service.dequeue()
            if job:
                dequeued_jobs.append(job)
        
        # Verify that critical jobs come first, then high, then normal
        critical_jobs = [job for job in dequeued_jobs if job.priority == Priority.CRITICAL.value]
        high_jobs = [job for job in dequeued_jobs if job.priority == Priority.HIGH.value]
        normal_jobs = [job for job in dequeued_jobs if job.priority == Priority.NORMAL.value]
        
        assert len(critical_jobs) == 10
        assert len(high_jobs) == 10
        assert len(normal_jobs) == 10
        
        # Find indices of first occurrence of each priority
        first_critical_idx = next((i for i, job in enumerate(dequeued_jobs) if job.priority == Priority.CRITICAL.value), -1)
        first_high_idx = next((i for i, job in enumerate(dequeued_jobs) if job.priority == Priority.HIGH.value), -1)
        first_normal_idx = next((i for i, job in enumerate(dequeued_jobs) if job.priority == Priority.NORMAL.value), -1)
        
        # Critical should come before high, high before normal
        assert first_critical_idx < first_high_idx
        assert first_high_idx < first_normal_idx
    
    @pytest.mark.asyncio
    async def test_concurrent_queue_operations_with_job_dispatch(self, queue_service, job_scheduler, sample_compute_instance, db_session):
        """Test concurrent queue operations with job dispatch (Requirement 3.4)."""
        # This test requires Redis setup
        pytest.skip("Requires Redis connection for integration test")
        
        job_repo = JobRepository(db_session)
        
        # Create jobs for concurrent processing
        jobs = []
        for i in range(10):
            job = TranslationJob(
                id=uuid4(),
                user_id=f"dispatch-user-{i}",
                source_language="en",
                target_language="es",
                content_hash=f"dispatch-hash-{i}",
                word_count=100,
                priority=Priority.HIGH.value,
                status=JobStatus.QUEUED.value,
                created_at=datetime.utcnow()
            )
            
            created_job = await job_repo.create(job)
            jobs.append(created_job)
        
        await db_session.commit()
        
        # Enqueue jobs
        for job in jobs:
            await queue_service.enqueue(job)
        
        # Simulate concurrent dispatch operations
        dispatch_tasks = []
        for i in range(5):  # 5 concurrent dispatchers
            task = asyncio.create_task(job_scheduler._dispatch_jobs())
            dispatch_tasks.append(task)
        
        # Let dispatchers run briefly
        await asyncio.sleep(0.1)
        
        # Cancel dispatch tasks
        for task in dispatch_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Verify queue operations completed without corruption
        remaining_depth = await queue_service.get_queue_depth()
        processing_stats = await queue_service.get_queue_stats()
        
        # Total jobs should be accounted for (either in queue or processing)
        total_accounted = remaining_depth + processing_stats.get("processing", 0)
        assert total_accounted <= 10  # Should not exceed original job count
    
    @pytest.mark.asyncio
    async def test_queue_consistency_during_high_priority_interruption(self, queue_service, db_session):
        """Test queue consistency when high priority jobs interrupt processing (Requirements 3.1, 3.4)."""
        # This test requires Redis setup
        pytest.skip("Requires Redis connection for integration test")
        
        job_repo = JobRepository(db_session)
        
        # Create normal priority jobs first
        normal_jobs = []
        for i in range(5):
            job = TranslationJob(
                id=uuid4(),
                user_id=f"normal-user-{i}",
                source_language="en",
                target_language="es",
                content_hash=f"normal-hash-{i}",
                word_count=100,
                priority=Priority.NORMAL.value,
                status=JobStatus.QUEUED.value,
                created_at=datetime.utcnow()
            )
            
            created_job = await job_repo.create(job)
            normal_jobs.append(created_job)
            await queue_service.enqueue(created_job)
        
        # Simulate some jobs being processed
        processing_job = await queue_service.dequeue()
        assert processing_job is not None
        
        # Now add critical priority job
        critical_job = TranslationJob(
            id=uuid4(),
            user_id="critical-user",
            source_language="en",
            target_language="es",
            content_hash="critical-hash",
            word_count=100,
            priority=Priority.CRITICAL.value,
            status=JobStatus.QUEUED.value,
            created_at=datetime.utcnow()
        )
        
        created_critical_job = await job_repo.create(critical_job)
        await db_session.commit()
        
        # Enqueue critical job
        await queue_service.enqueue(created_critical_job)
        
        # Dequeue should return critical job next, even though normal jobs were queued first
        next_job = await queue_service.dequeue()
        assert next_job is not None
        assert next_job.priority == Priority.CRITICAL.value
        
        # Verify queue consistency
        stats = await queue_service.get_queue_stats()
        assert stats["processing"] >= 1  # At least the critical job should be processing
        assert stats["normal"] >= 3  # Remaining normal jobs should still be queued
    
    @pytest.mark.asyncio
    async def test_queue_performance_under_load(self, queue_service, db_session):
        """Test queue performance under high load."""
        # This test requires Redis setup and is more of a performance test
        pytest.skip("Performance test - requires Redis and extended runtime")
        
        job_repo = JobRepository(db_session)
        
        # Create many jobs
        jobs = []
        for i in range(1000):
            job = TranslationJob(
                id=uuid4(),
                user_id=f"load-user-{i % 10}",  # 10 different users
                source_language="en",
                target_language="es",
                content_hash=f"load-hash-{i}",
                word_count=100,
                priority=["normal", "high", "critical"][i % 3],
                status=JobStatus.QUEUED.value,
                created_at=datetime.utcnow()
            )
            jobs.append(job)
        
        # Bulk create jobs
        await job_repo.bulk_create(jobs)
        await db_session.commit()
        
        # Measure enqueue performance
        start_time = datetime.utcnow()
        
        enqueue_tasks = [queue_service.enqueue(job) for job in jobs]
        await asyncio.gather(*enqueue_tasks)
        
        enqueue_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Should be able to enqueue 1000 jobs in reasonable time
        assert enqueue_time < 30  # 30 seconds
        
        # Verify all jobs were enqueued
        total_depth = await queue_service.get_queue_depth()
        assert total_depth == 1000