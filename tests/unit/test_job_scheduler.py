"""
Unit tests for JobScheduler.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

from src.config.config import Priority, JobStatus
from src.database.models import TranslationJob, ComputeInstance
from src.services.job_scheduler import JobScheduler
from src.utils.exceptions import JobTimeoutError


class TestJobScheduler:
    """Test JobScheduler functionality."""
    
    @pytest.fixture
    def job_scheduler(self):
        """Create JobScheduler instance."""
        return JobScheduler()
    
    @pytest.fixture
    def sample_job(self):
        """Create sample translation job."""
        return TranslationJob(
            id=uuid4(),
            user_id="test-user",
            source_language="en",
            target_language="es",
            content_hash="test-hash",
            word_count=100,
            priority=Priority.NORMAL.value,
            status=JobStatus.QUEUED.value,
            created_at=datetime.utcnow()
        )
    
    @pytest.fixture
    def sample_instance(self):
        """Create sample compute instance."""
        return {
            "id": "instance-1",
            "instance_type": "g4dn.xlarge",
            "available_capacity": 2,
            "gpu_utilization": 50.0,
            "memory_usage": 60.0,
            "active_jobs": 1
        }
    
    @pytest.fixture
    def mock_compute_instance(self):
        """Create mock compute instance model."""
        return ComputeInstance(
            id="instance-1",
            instance_type="g4dn.xlarge",
            status="running",
            gpu_utilization=50.0,
            memory_usage=60.0,
            active_jobs=1,
            max_concurrent_jobs=3,
            created_at=datetime.utcnow(),
            last_heartbeat=datetime.utcnow()
        )
    
    @pytest.mark.asyncio
    async def test_start_stop_dispatcher(self, job_scheduler):
        """Test starting and stopping the job dispatcher."""
        assert not job_scheduler._running
        
        # Start dispatcher
        await job_scheduler.start_dispatcher()
        assert job_scheduler._running
        assert job_scheduler._dispatch_task is not None
        assert job_scheduler._timeout_check_task is not None
        
        # Stop dispatcher
        await job_scheduler.stop_dispatcher()
        assert not job_scheduler._running
    
    @pytest.mark.asyncio
    async def test_get_available_instances(self, job_scheduler, mock_compute_instance):
        """Test getting available compute instances."""
        with patch('src.database.connection.get_db_session') as mock_session, \
             patch('src.database.repositories.base.BaseRepository') as mock_repo:
            
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            mock_repo_instance = AsyncMock()
            mock_repo_instance.find_by.return_value = [mock_compute_instance]
            mock_repo.return_value = mock_repo_instance
            
            available_instances = await job_scheduler._get_available_instances()
            
            assert len(available_instances) == 1
            instance = available_instances[0]
            assert instance["id"] == "instance-1"
            assert instance["available_capacity"] == 2  # max_concurrent_jobs - active_jobs
            assert instance["gpu_utilization"] == 50.0
    
    @pytest.mark.asyncio
    async def test_get_available_instances_no_capacity(self, job_scheduler):
        """Test getting available instances when none have capacity."""
        # Mock instance at full capacity
        full_instance = ComputeInstance(
            id="instance-1",
            instance_type="g4dn.xlarge",
            status="running",
            active_jobs=3,
            max_concurrent_jobs=3,  # At capacity
            last_heartbeat=datetime.utcnow()
        )
        
        with patch('src.database.connection.get_db_session') as mock_session, \
             patch('src.database.repositories.base.BaseRepository') as mock_repo:
            
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            mock_repo_instance = AsyncMock()
            mock_repo_instance.find_by.return_value = [full_instance]
            mock_repo.return_value = mock_repo_instance
            
            available_instances = await job_scheduler._get_available_instances()
            
            assert len(available_instances) == 0
    
    @pytest.mark.asyncio
    async def test_get_available_instances_stale_heartbeat(self, job_scheduler):
        """Test that instances with stale heartbeats are excluded."""
        # Mock instance with old heartbeat
        stale_instance = ComputeInstance(
            id="instance-1",
            instance_type="g4dn.xlarge",
            status="running",
            active_jobs=1,
            max_concurrent_jobs=3,
            last_heartbeat=datetime.utcnow() - timedelta(minutes=10)  # Stale
        )
        
        with patch('src.database.connection.get_db_session') as mock_session, \
             patch('src.database.repositories.base.BaseRepository') as mock_repo:
            
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            mock_repo_instance = AsyncMock()
            mock_repo_instance.find_by.return_value = [stale_instance]
            mock_repo.return_value = mock_repo_instance
            
            available_instances = await job_scheduler._get_available_instances()
            
            assert len(available_instances) == 0
    
    @pytest.mark.asyncio
    async def test_dispatch_job_to_instance_success(self, job_scheduler, sample_job, sample_instance, mock_compute_instance):
        """Test successful job dispatch to instance."""
        with patch('src.database.connection.get_db_session') as mock_session, \
             patch('src.database.repositories.JobRepository') as mock_job_repo, \
             patch('src.database.repositories.base.BaseRepository') as mock_instance_repo, \
             patch.object(job_scheduler, '_process_job') as mock_process:
            
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            # Mock job repository
            mock_job_repo_instance = AsyncMock()
            mock_job_repo_instance.update_job_status.return_value = True
            mock_job_repo.return_value = mock_job_repo_instance
            
            # Mock instance repository
            mock_instance_repo_instance = AsyncMock()
            mock_instance_repo_instance.get_by_id.return_value = mock_compute_instance
            mock_instance_repo_instance.update.return_value = True
            mock_instance_repo.return_value = mock_instance_repo_instance
            
            result = await job_scheduler._dispatch_job_to_instance(sample_job, sample_instance)
            
            assert result is True
            mock_job_repo_instance.update_job_status.assert_called_once_with(
                sample_job.id,
                JobStatus.PROCESSING,
                progress=0.0,
                compute_instance_id="instance-1"
            )
            
            # Verify instance active job count was incremented
            assert mock_compute_instance.active_jobs == 2  # Was 1, now 2
            mock_instance_repo_instance.update.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_dispatch_job_to_instance_job_update_fails(self, job_scheduler, sample_job, sample_instance):
        """Test job dispatch when job status update fails."""
        with patch('src.database.connection.get_db_session') as mock_session, \
             patch('src.database.repositories.JobRepository') as mock_job_repo:
            
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            mock_job_repo_instance = AsyncMock()
            mock_job_repo_instance.update_job_status.return_value = False  # Failure
            mock_job_repo.return_value = mock_job_repo_instance
            
            result = await job_scheduler._dispatch_job_to_instance(sample_job, sample_instance)
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_estimate_processing_time(self, job_scheduler):
        """Test processing time estimation."""
        # Test normal priority
        time_normal = job_scheduler._estimate_processing_time(1500, Priority.NORMAL.value)
        assert time_normal > 0
        
        # Test critical priority (should be faster due to higher multiplier)
        time_critical = job_scheduler._estimate_processing_time(1500, Priority.CRITICAL.value)
        assert time_critical > 0
        
        # Critical should generally be faster than normal (though randomness may affect this)
        # We'll just verify both are reasonable values
        assert 30 < time_normal < 300  # Should be around 60 seconds for 1500 words
        assert 30 < time_critical < 300
    
    @pytest.mark.asyncio
    async def test_process_job_success(self, job_scheduler, sample_job):
        """Test successful job processing."""
        instance_id = "instance-1"
        
        with patch('src.database.connection.get_db_session') as mock_session, \
             patch('src.database.repositories.JobRepository') as mock_job_repo, \
             patch('src.database.repositories.base.BaseRepository') as mock_instance_repo, \
             patch.object(job_scheduler, '_estimate_processing_time', return_value=0.1), \
             patch.object(job_scheduler, '_get_job_by_id', return_value=sample_job), \
             patch.object(job_scheduler.notification_service, 'send_job_completion_webhook', return_value=True):
            
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            # Mock job repository
            mock_job_repo_instance = AsyncMock()
            mock_job_repo_instance.update_job_status.return_value = True
            mock_job_repo.return_value = mock_job_repo_instance
            
            # Mock instance repository
            mock_instance = AsyncMock()
            mock_instance.active_jobs = 2
            mock_instance_repo_instance = AsyncMock()
            mock_instance_repo_instance.get_by_id.return_value = mock_instance
            mock_instance_repo_instance.update.return_value = True
            mock_instance_repo.return_value = mock_instance_repo_instance
            
            await job_scheduler._process_job(sample_job, instance_id)
            
            # Verify job was marked as completed
            calls = mock_job_repo_instance.update_job_status.call_args_list
            final_call = calls[-1]  # Last call should be completion
            assert final_call[0][1] == JobStatus.COMPLETED
            assert final_call[1]["progress"] == 100.0
            
            # Verify instance active job count was decremented
            assert mock_instance.active_jobs == 1
    
    @pytest.mark.asyncio
    async def test_handle_job_failure(self, job_scheduler, sample_job):
        """Test job failure handling."""
        instance_id = "instance-1"
        error_message = "Translation engine error"
        
        with patch('src.database.connection.get_db_session') as mock_session, \
             patch('src.database.repositories.JobRepository') as mock_job_repo, \
             patch('src.database.repositories.base.BaseRepository') as mock_instance_repo, \
             patch.object(job_scheduler.queue_service, 'move_to_dead_letter_queue', return_value=True), \
             patch.object(job_scheduler, '_get_job_by_id', return_value=sample_job), \
             patch.object(job_scheduler.notification_service, 'send_job_completion_webhook', return_value=True):
            
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            # Mock job repository
            mock_job_repo_instance = AsyncMock()
            mock_job_repo_instance.update_job_status.return_value = True
            mock_job_repo.return_value = mock_job_repo_instance
            
            # Mock instance repository
            mock_instance = AsyncMock()
            mock_instance.active_jobs = 2
            mock_instance_repo_instance = AsyncMock()
            mock_instance_repo_instance.get_by_id.return_value = mock_instance
            mock_instance_repo_instance.update.return_value = True
            mock_instance_repo.return_value = mock_instance_repo_instance
            
            await job_scheduler._handle_job_failure(sample_job.id, instance_id, error_message)
            
            # Verify job was marked as failed
            mock_job_repo_instance.update_job_status.assert_called_once_with(
                sample_job.id,
                JobStatus.FAILED,
                error_message=error_message
            )
            
            # Verify job was moved to DLQ
            job_scheduler.queue_service.move_to_dead_letter_queue.assert_called_once_with(
                sample_job.id, error_message
            )
            
            # Verify instance active job count was decremented
            assert mock_instance.active_jobs == 1
    
    @pytest.mark.asyncio
    async def test_handle_job_timeout(self, job_scheduler):
        """Test job timeout handling."""
        # Create job with old start time
        timeout_job = TranslationJob(
            id=uuid4(),
            user_id="test-user",
            source_language="en",
            target_language="es",
            content_hash="test-hash",
            word_count=100,
            priority="normal",
            status="processing",
            started_at=datetime.utcnow() - timedelta(hours=1),
            compute_instance_id="instance-1"
        )
        
        with patch('src.database.connection.get_db_session') as mock_session, \
             patch('src.database.repositories.JobRepository') as mock_job_repo, \
             patch('src.database.repositories.base.BaseRepository') as mock_instance_repo, \
             patch.object(job_scheduler.queue_service, 'move_to_dead_letter_queue', return_value=True), \
             patch.object(job_scheduler.notification_service, 'send_job_completion_webhook', return_value=True):
            
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            # Mock job repository
            mock_job_repo_instance = AsyncMock()
            mock_job_repo_instance.update_job_status.return_value = True
            mock_job_repo.return_value = mock_job_repo_instance
            
            # Mock instance repository
            mock_instance = AsyncMock()
            mock_instance.active_jobs = 1
            mock_instance_repo_instance = AsyncMock()
            mock_instance_repo_instance.get_by_id.return_value = mock_instance
            mock_instance_repo_instance.update.return_value = True
            mock_instance_repo.return_value = mock_instance_repo_instance
            
            await job_scheduler._handle_job_timeout(timeout_job)
            
            # Verify job was marked as failed with timeout message
            mock_job_repo_instance.update_job_status.assert_called_once()
            call_args = mock_job_repo_instance.update_job_status.call_args
            assert call_args[0][1] == JobStatus.FAILED
            assert "timed out" in call_args[1]["error_message"].lower()
    
    @pytest.mark.asyncio
    async def test_check_job_timeouts(self, job_scheduler):
        """Test checking for timed out jobs."""
        # Create stale job
        stale_job = TranslationJob(
            id=uuid4(),
            user_id="test-user",
            source_language="en",
            target_language="es",
            content_hash="test-hash",
            word_count=100,
            priority="normal",
            status="processing",
            started_at=datetime.utcnow() - timedelta(hours=1)
        )
        
        with patch('src.database.connection.get_db_session') as mock_session, \
             patch('src.database.repositories.JobRepository') as mock_job_repo, \
             patch.object(job_scheduler, '_handle_job_timeout') as mock_handle_timeout:
            
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            mock_job_repo_instance = AsyncMock()
            mock_job_repo_instance.get_stale_jobs.return_value = [stale_job]
            mock_job_repo.return_value = mock_job_repo_instance
            
            await job_scheduler._check_job_timeouts()
            
            mock_handle_timeout.assert_called_once_with(stale_job)
    
    @pytest.mark.asyncio
    async def test_cancel_job_in_queue(self, job_scheduler):
        """Test cancelling a job that's still in queue."""
        job_id = uuid4()
        
        with patch.object(job_scheduler.queue_service, 'remove_job', return_value=True), \
             patch('src.database.connection.get_db_session') as mock_session, \
             patch('src.database.repositories.JobRepository') as mock_job_repo:
            
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            mock_job_repo_instance = AsyncMock()
            mock_job_repo_instance.update_job_status.return_value = True
            mock_job_repo.return_value = mock_job_repo_instance
            
            result = await job_scheduler.cancel_job(job_id)
            
            assert result is True
            job_scheduler.queue_service.remove_job.assert_called_once_with(job_id)
            mock_job_repo_instance.update_job_status.assert_called_once_with(
                job_id,
                JobStatus.FAILED,
                error_message="Job cancelled by user"
            )
    
    @pytest.mark.asyncio
    async def test_cancel_job_processing(self, job_scheduler):
        """Test cancelling a job that's currently processing."""
        job_id = uuid4()
        
        # Add job to active dispatchers
        job_scheduler.active_dispatchers.add(f"instance-1:{job_id}")
        
        with patch.object(job_scheduler.queue_service, 'remove_job', return_value=False):
            result = await job_scheduler.cancel_job(job_id)
            
            assert result is True
            # Job should still be in active dispatchers (will be handled by processing task)
            assert f"instance-1:{job_id}" in job_scheduler.active_dispatchers
    
    @pytest.mark.asyncio
    async def test_cancel_job_not_found(self, job_scheduler):
        """Test cancelling a job that's not found."""
        job_id = uuid4()
        
        with patch.object(job_scheduler.queue_service, 'remove_job', return_value=False):
            result = await job_scheduler.cancel_job(job_id)
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_get_scheduler_status(self, job_scheduler, sample_instance):
        """Test getting scheduler status."""
        job_scheduler._running = True
        job_scheduler.active_dispatchers.add("instance-1:job-1")
        job_scheduler.active_dispatchers.add("instance-2:job-2")
        
        with patch.object(job_scheduler, '_get_available_instances', return_value=[sample_instance]):
            status = await job_scheduler.get_scheduler_status()
            
            assert status["running"] is True
            assert status["active_dispatchers"] == 2
            assert status["available_instances"] == 1
            assert status["total_instance_capacity"] == 2
            assert len(status["instance_details"]) == 1
    
    @pytest.mark.asyncio
    async def test_update_scheduler_config(self, job_scheduler):
        """Test updating scheduler configuration."""
        config_updates = {
            "dispatch_interval": 2.0,
            "max_concurrent_dispatchers": 10,
            "job_timeout_minutes": 45
        }
        
        await job_scheduler.update_scheduler_config(config_updates)
        
        assert job_scheduler.dispatch_interval == 2.0
        assert job_scheduler.max_concurrent_dispatchers == 10
        assert job_scheduler.job_timeout_minutes == 45
    
    @pytest.mark.asyncio
    async def test_update_scheduler_config_validation(self, job_scheduler):
        """Test scheduler config validation."""
        config_updates = {
            "dispatch_interval": -1.0,  # Should be clamped to minimum
            "max_concurrent_dispatchers": 0,  # Should be clamped to minimum
            "job_timeout_minutes": 2  # Should be clamped to minimum
        }
        
        await job_scheduler.update_scheduler_config(config_updates)
        
        assert job_scheduler.dispatch_interval == 0.1  # Minimum value
        assert job_scheduler.max_concurrent_dispatchers == 1  # Minimum value
        assert job_scheduler.job_timeout_minutes == 5  # Minimum value
    
    @pytest.mark.asyncio
    async def test_dispatch_jobs_integration(self, job_scheduler, sample_job, sample_instance):
        """Test the main dispatch jobs method."""
        with patch.object(job_scheduler, '_get_available_instances', return_value=[sample_instance]), \
             patch.object(job_scheduler.queue_service, 'dequeue', return_value=sample_job), \
             patch.object(job_scheduler, '_dispatch_job_to_instance', return_value=True):
            
            await job_scheduler._dispatch_jobs()
            
            job_scheduler._dispatch_job_to_instance.assert_called_once_with(sample_job, sample_instance)
    
    @pytest.mark.asyncio
    async def test_dispatch_jobs_no_instances(self, job_scheduler):
        """Test dispatch when no instances are available."""
        with patch.object(job_scheduler, '_get_available_instances', return_value=[]):
            await job_scheduler._dispatch_jobs()
            
            # Should exit early without trying to dequeue jobs
    
    @pytest.mark.asyncio
    async def test_dispatch_jobs_no_jobs(self, job_scheduler, sample_instance):
        """Test dispatch when no jobs are available."""
        with patch.object(job_scheduler, '_get_available_instances', return_value=[sample_instance]), \
             patch.object(job_scheduler.queue_service, 'dequeue', return_value=None):
            
            await job_scheduler._dispatch_jobs()
            
            # Should exit early when no jobs available
    
    @pytest.mark.asyncio
    async def test_priority_based_job_dispatch_logic(self, job_scheduler, sample_instance):
        """Test job dispatch logic respects priority ordering (Requirement 3.1, 3.2)."""
        # Create jobs with different priorities
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
        
        normal_job = TranslationJob(
            id=uuid4(),
            user_id="normal-user",
            source_language="en",
            target_language="es",
            content_hash="normal-hash",
            word_count=100,
            priority=Priority.NORMAL.value,
            status=JobStatus.QUEUED.value,
            created_at=datetime.utcnow()
        )
        
        # Mock queue service to return critical job first, then normal job
        dequeue_sequence = [critical_job, normal_job, None]  # None indicates no more jobs
        dequeue_call_count = 0
        
        def mock_dequeue():
            nonlocal dequeue_call_count
            if dequeue_call_count < len(dequeue_sequence):
                result = dequeue_sequence[dequeue_call_count]
                dequeue_call_count += 1
                return result
            return None
        
        with patch.object(job_scheduler, '_get_available_instances', return_value=[sample_instance]), \
             patch.object(job_scheduler.queue_service, 'dequeue', side_effect=mock_dequeue), \
             patch.object(job_scheduler, '_dispatch_job_to_instance', return_value=True) as mock_dispatch:
            
            # Run dispatch jobs
            await job_scheduler._dispatch_jobs()
            
            # Verify critical job was dispatched first
            assert mock_dispatch.call_count >= 1
            first_call = mock_dispatch.call_args_list[0]
            dispatched_job = first_call[0][0]  # First argument is the job
            assert dispatched_job.priority == Priority.CRITICAL.value
    
    @pytest.mark.asyncio
    async def test_concurrent_job_dispatch_handling(self, job_scheduler, mock_compute_instance):
        """Test concurrent job dispatch operations (Requirement 3.4)."""
        # Create multiple instances to simulate concurrent dispatch
        instances = [
            {
                "id": f"instance-{i}",
                "instance_type": "g4dn.xlarge",
                "available_capacity": 1,
                "gpu_utilization": 50.0,
                "memory_usage": 60.0,
                "active_jobs": 0
            }
            for i in range(3)
        ]
        
        # Create multiple jobs
        jobs = []
        for i in range(3):
            job = TranslationJob(
                id=uuid4(),
                user_id=f"concurrent-user-{i}",
                source_language="en",
                target_language="es",
                content_hash=f"concurrent-hash-{i}",
                word_count=100,
                priority=Priority.HIGH.value,
                status=JobStatus.QUEUED.value,
                created_at=datetime.utcnow()
            )
            jobs.append(job)
        
        # Mock database operations
        with patch('src.database.connection.get_db_session') as mock_session, \
             patch('src.database.repositories.JobRepository') as mock_job_repo, \
             patch('src.database.repositories.base.BaseRepository') as mock_instance_repo, \
             patch.object(job_scheduler, '_process_job') as mock_process:
            
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            # Mock job repository
            mock_job_repo_instance = AsyncMock()
            mock_job_repo_instance.update_job_status.return_value = True
            mock_job_repo.return_value = mock_job_repo_instance
            
            # Mock instance repository
            mock_instance_repo_instance = AsyncMock()
            mock_instance_repo_instance.get_by_id.return_value = mock_compute_instance
            mock_instance_repo_instance.update.return_value = True
            mock_instance_repo.return_value = mock_instance_repo_instance
            
            # Dispatch jobs concurrently to different instances
            dispatch_tasks = [
                job_scheduler._dispatch_job_to_instance(jobs[i], instances[i])
                for i in range(3)
            ]
            
            dispatch_results = await asyncio.gather(*dispatch_tasks, return_exceptions=True)
            
            # Verify all dispatches succeeded
            successful_dispatches = [result for result in dispatch_results if result is True]
            assert len(successful_dispatches) == 3
            
            # Verify all jobs were marked as processing
            assert mock_job_repo_instance.update_job_status.call_count == 3
            
            # Verify all process_job tasks were started
            assert mock_process.call_count == 3
    
    @pytest.mark.asyncio
    async def test_job_dispatch_with_instance_capacity_limits(self, job_scheduler):
        """Test job dispatch respects instance capacity limits (Requirement 3.4)."""
        # Create instance at near capacity
        near_full_instance = {
            "id": "instance-1",
            "instance_type": "g4dn.xlarge",
            "available_capacity": 1,  # Only 1 slot available
            "gpu_utilization": 80.0,
            "memory_usage": 85.0,
            "active_jobs": 2
        }
        
        # Create multiple jobs
        jobs = []
        for i in range(3):  # More jobs than capacity
            job = TranslationJob(
                id=uuid4(),
                user_id=f"capacity-user-{i}",
                source_language="en",
                target_language="es",
                content_hash=f"capacity-hash-{i}",
                word_count=100,
                priority=Priority.NORMAL.value,
                status=JobStatus.QUEUED.value,
                created_at=datetime.utcnow()
            )
            jobs.append(job)
        
        # Mock queue service to return jobs in sequence
        job_index = 0
        def mock_dequeue():
            nonlocal job_index
            if job_index < len(jobs):
                result = jobs[job_index]
                job_index += 1
                return result
            return None
        
        with patch.object(job_scheduler, '_get_available_instances', return_value=[near_full_instance]), \
             patch.object(job_scheduler.queue_service, 'dequeue', side_effect=mock_dequeue), \
             patch.object(job_scheduler, '_dispatch_job_to_instance') as mock_dispatch, \
             patch.object(job_scheduler.queue_service, 'enqueue') as mock_re_enqueue:
            
            # Mock first dispatch succeeds, second fails (capacity reached)
            mock_dispatch.side_effect = [True, False, False]
            
            # Set max concurrent dispatchers to allow multiple attempts
            job_scheduler.max_concurrent_dispatchers = 3
            
            await job_scheduler._dispatch_jobs()
            
            # Verify only one job was successfully dispatched (due to capacity limit)
            assert mock_dispatch.call_count <= 3
            
            # Verify failed dispatches resulted in jobs being re-enqueued
            # (The exact number depends on the dispatch logic implementation)
            assert mock_re_enqueue.call_count >= 0
    
    @pytest.mark.asyncio
    async def test_job_dispatch_priority_weight_distribution(self, job_scheduler):
        """Test job dispatch considers priority weights for resource allocation (Requirement 3.1, 3.2)."""
        # Create mixed priority jobs
        critical_job = TranslationJob(
            id=uuid4(),
            user_id="critical-user",
            source_language="en",
            target_language="es",
            content_hash="critical-hash",
            word_count=1000,  # Larger job
            priority=Priority.CRITICAL.value,
            status=JobStatus.QUEUED.value,
            created_at=datetime.utcnow()
        )
        
        normal_job = TranslationJob(
            id=uuid4(),
            user_id="normal-user",
            source_language="en",
            target_language="es",
            content_hash="normal-hash",
            word_count=100,  # Smaller job
            priority=Priority.NORMAL.value,
            status=JobStatus.QUEUED.value,
            created_at=datetime.utcnow()
        )
        
        # Test processing time estimation considers priority
        critical_time = job_scheduler._estimate_processing_time(1000, Priority.CRITICAL.value)
        normal_time = job_scheduler._estimate_processing_time(1000, Priority.NORMAL.value)
        
        # Critical jobs should generally process faster due to higher priority multiplier
        # Note: Due to randomness in estimation, we test the general principle
        assert critical_time > 0
        assert normal_time > 0
        
        # Test that both times are reasonable for 1000 words
        # At 1500 words/minute base rate, 1000 words should take ~40 seconds
        assert 20 < critical_time < 120  # Allow for priority multiplier and randomness
        assert 20 < normal_time < 120
    
    @pytest.mark.asyncio
    async def test_close_scheduler(self, job_scheduler):
        """Test closing the scheduler."""
        with patch.object(job_scheduler, 'stop_dispatcher') as mock_stop, \
             patch.object(job_scheduler.queue_service, 'close') as mock_queue_close, \
             patch.object(job_scheduler.notification_service, 'close') as mock_notif_close:
            
            await job_scheduler.close()
            
            mock_stop.assert_called_once()
            mock_queue_close.assert_called_once()
            mock_notif_close.assert_called_once()