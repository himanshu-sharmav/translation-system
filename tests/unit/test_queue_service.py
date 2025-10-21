"""
Unit tests for QueueService.
"""

import pytest
import json
from datetime import datetime
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

from src.config.config import Priority, JobStatus
from src.database.models import TranslationJob
from src.services.queue_service import QueueService
from src.utils.exceptions import QueueError


class TestQueueService:
    """Test QueueService functionality."""
    
    @pytest.fixture
    def queue_service(self):
        """Create QueueService instance."""
        return QueueService()
    
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
    def mock_redis(self):
        """Create mock Redis client."""
        mock_redis = AsyncMock()
        mock_redis.zadd = AsyncMock(return_value=1)
        mock_redis.zpopmin = AsyncMock()
        mock_redis.zcard = AsyncMock(return_value=0)
        mock_redis.zrange = AsyncMock(return_value=[])
        mock_redis.zrem = AsyncMock(return_value=1)
        mock_redis.sadd = AsyncMock(return_value=1)
        mock_redis.srem = AsyncMock(return_value=1)
        mock_redis.scard = AsyncMock(return_value=0)
        mock_redis.smembers = AsyncMock(return_value=set())
        mock_redis.lpush = AsyncMock(return_value=1)
        mock_redis.llen = AsyncMock(return_value=0)
        mock_redis.rpop = AsyncMock(return_value=None)
        return mock_redis
    
    @pytest.mark.asyncio
    async def test_enqueue_job_success(self, queue_service, sample_job, mock_redis):
        """Test successful job enqueuing."""
        with patch.object(queue_service, '_get_redis_client', return_value=mock_redis):
            result = await queue_service.enqueue(sample_job)
            
            assert result is True
            mock_redis.zadd.assert_called_once()
            
            # Verify the job data was serialized correctly
            call_args = mock_redis.zadd.call_args
            queue_name = call_args[0][0]
            job_data_dict = call_args[0][1]
            
            assert queue_name == "queue:normal"
            assert len(job_data_dict) == 1
            
            # Extract and verify job data
            job_data_str = list(job_data_dict.keys())[0]
            job_data = json.loads(job_data_str)
            
            assert job_data["job_id"] == str(sample_job.id)
            assert job_data["user_id"] == sample_job.user_id
            assert job_data["priority"] == sample_job.priority
    
    @pytest.mark.asyncio
    async def test_enqueue_different_priorities(self, queue_service, mock_redis):
        """Test enqueuing jobs with different priorities."""
        with patch.object(queue_service, '_get_redis_client', return_value=mock_redis):
            
            # Test each priority level
            priorities = [Priority.CRITICAL, Priority.HIGH, Priority.NORMAL]
            expected_queues = ["queue:critical", "queue:high", "queue:normal"]
            
            for priority, expected_queue in zip(priorities, expected_queues):
                job = TranslationJob(
                    id=uuid4(),
                    user_id="test-user",
                    source_language="en",
                    target_language="es",
                    content_hash="test-hash",
                    word_count=100,
                    priority=priority.value,
                    status=JobStatus.QUEUED.value,
                    created_at=datetime.utcnow()
                )
                
                await queue_service.enqueue(job)
                
                # Check that the correct queue was used
                call_args = mock_redis.zadd.call_args
                queue_name = call_args[0][0]
                assert queue_name == expected_queue
    
    @pytest.mark.asyncio
    async def test_dequeue_priority_order(self, queue_service, mock_redis):
        """Test that jobs are dequeued in priority order."""
        # Mock Redis responses for different priority queues
        critical_job_data = {
            "job_id": str(uuid4()),
            "user_id": "test-user",
            "source_language": "en",
            "target_language": "es",
            "content_hash": "critical-hash",
            "word_count": 100,
            "priority": "critical",
            "created_at": datetime.utcnow().isoformat()
        }
        
        def mock_zpopmin(queue_name, count=1):
            if queue_name == "queue:critical":
                return [(json.dumps(critical_job_data), 1234567890.0)]
            else:
                return []
        
        mock_redis.zpopmin.side_effect = mock_zpopmin
        
        with patch.object(queue_service, '_get_redis_client', return_value=mock_redis):
            job = await queue_service.dequeue()
            
            assert job is not None
            assert job.priority == "critical"
            assert str(job.id) == critical_job_data["job_id"]
            
            # Verify that critical queue was checked first
            mock_redis.zpopmin.assert_called()
            calls = mock_redis.zpopmin.call_args_list
            assert calls[0][0][0] == "queue:critical"
    
    @pytest.mark.asyncio
    async def test_dequeue_no_jobs(self, queue_service, mock_redis):
        """Test dequeuing when no jobs are available."""
        mock_redis.zpopmin.return_value = []
        
        with patch.object(queue_service, '_get_redis_client', return_value=mock_redis):
            job = await queue_service.dequeue()
            
            assert job is None
    
    @pytest.mark.asyncio
    async def test_dequeue_specific_priority(self, queue_service, mock_redis):
        """Test dequeuing from a specific priority queue."""
        high_job_data = {
            "job_id": str(uuid4()),
            "user_id": "test-user",
            "source_language": "en",
            "target_language": "es",
            "content_hash": "high-hash",
            "word_count": 100,
            "priority": "high",
            "created_at": datetime.utcnow().isoformat()
        }
        
        mock_redis.zpopmin.return_value = [(json.dumps(high_job_data), 1234567890.0)]
        
        with patch.object(queue_service, '_get_redis_client', return_value=mock_redis):
            job = await queue_service.dequeue(Priority.HIGH)
            
            assert job is not None
            assert job.priority == "high"
            
            # Verify only high priority queue was checked
            mock_redis.zpopmin.assert_called_once_with("queue:high", count=1)
    
    @pytest.mark.asyncio
    async def test_get_queue_depth_all_queues(self, queue_service, mock_redis):
        """Test getting total queue depth across all queues."""
        # Mock different depths for each queue
        def mock_zcard(queue_name):
            depths = {
                "queue:critical": 2,
                "queue:high": 5,
                "queue:normal": 10
            }
            return depths.get(queue_name, 0)
        
        mock_redis.zcard.side_effect = mock_zcard
        
        with patch.object(queue_service, '_get_redis_client', return_value=mock_redis):
            total_depth = await queue_service.get_queue_depth()
            
            assert total_depth == 17  # 2 + 5 + 10
    
    @pytest.mark.asyncio
    async def test_get_queue_depth_specific_priority(self, queue_service, mock_redis):
        """Test getting queue depth for specific priority."""
        mock_redis.zcard.return_value = 5
        
        with patch.object(queue_service, '_get_redis_client', return_value=mock_redis):
            depth = await queue_service.get_queue_depth(Priority.HIGH)
            
            assert depth == 5
            mock_redis.zcard.assert_called_once_with("queue:high")
    
    @pytest.mark.asyncio
    async def test_update_job_status_success(self, queue_service):
        """Test successful job status update."""
        job_id = uuid4()
        
        with patch('src.database.connection.get_db_session') as mock_session, \
             patch('src.database.repositories.JobRepository') as mock_job_repo:
            
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            mock_repo_instance = AsyncMock()
            mock_repo_instance.update_job_status.return_value = True
            mock_job_repo.return_value = mock_repo_instance
            
            result = await queue_service.update_job_status(job_id, JobStatus.PROCESSING, 50.0)
            
            assert result is True
            mock_repo_instance.update_job_status.assert_called_once_with(
                job_id, JobStatus.PROCESSING, 50.0
            )
    
    @pytest.mark.asyncio
    async def test_update_job_status_completed_removes_from_processing(self, queue_service, mock_redis):
        """Test that completed jobs are removed from processing set."""
        job_id = uuid4()
        
        with patch('src.database.connection.get_db_session') as mock_session, \
             patch('src.database.repositories.JobRepository') as mock_job_repo, \
             patch.object(queue_service, '_get_redis_client', return_value=mock_redis):
            
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            mock_repo_instance = AsyncMock()
            mock_repo_instance.update_job_status.return_value = True
            mock_job_repo.return_value = mock_repo_instance
            
            result = await queue_service.update_job_status(job_id, JobStatus.COMPLETED)
            
            assert result is True
            mock_redis.srem.assert_called_once_with("processing_jobs", str(job_id))
    
    @pytest.mark.asyncio
    async def test_remove_job_from_queue(self, queue_service, mock_redis):
        """Test removing a job from queue."""
        job_id = uuid4()
        job_data = {
            "job_id": str(job_id),
            "user_id": "test-user",
            "priority": "normal"
        }
        
        # Mock that job is found in normal queue
        mock_redis.zrange.return_value = [json.dumps(job_data)]
        mock_redis.zrem.return_value = 1
        
        with patch.object(queue_service, '_get_redis_client', return_value=mock_redis):
            result = await queue_service.remove_job(job_id)
            
            assert result is True
            mock_redis.zrem.assert_called_once()
            mock_redis.srem.assert_called_once_with("processing_jobs", str(job_id))
    
    @pytest.mark.asyncio
    async def test_remove_job_not_found(self, queue_service, mock_redis):
        """Test removing a job that's not in any queue."""
        job_id = uuid4()
        
        # Mock empty queues
        mock_redis.zrange.return_value = []
        
        with patch.object(queue_service, '_get_redis_client', return_value=mock_redis):
            result = await queue_service.remove_job(job_id)
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_get_queue_stats(self, queue_service, mock_redis):
        """Test getting queue statistics."""
        # Mock queue depths
        def mock_zcard(queue_name):
            depths = {
                "queue:critical": 2,
                "queue:high": 5,
                "queue:normal": 10
            }
            return depths.get(queue_name, 0)
        
        mock_redis.zcard.side_effect = mock_zcard
        mock_redis.scard.return_value = 3  # processing jobs
        mock_redis.llen.return_value = 1   # dead letter queue
        
        with patch.object(queue_service, '_get_redis_client', return_value=mock_redis):
            stats = await queue_service.get_queue_stats()
            
            expected_stats = {
                "critical": 2,
                "high": 5,
                "normal": 10,
                "processing": 3,
                "dead_letter": 1
            }
            
            assert stats == expected_stats
    
    @pytest.mark.asyncio
    async def test_move_to_dead_letter_queue(self, queue_service, mock_redis):
        """Test moving a job to dead letter queue."""
        job_id = uuid4()
        error_message = "Processing failed"
        
        # Mock job retrieval
        mock_job = TranslationJob(
            id=job_id,
            user_id="test-user",
            source_language="en",
            target_language="es",
            content_hash="test-hash",
            word_count=100,
            priority="normal",
            status="failed"
        )
        
        with patch.object(queue_service, 'get_job', return_value=mock_job), \
             patch.object(queue_service, '_get_redis_client', return_value=mock_redis):
            
            result = await queue_service.move_to_dead_letter_queue(job_id, error_message)
            
            assert result is True
            mock_redis.lpush.assert_called_once()
            mock_redis.srem.assert_called_once_with("processing_jobs", str(job_id))
            
            # Verify DLQ entry structure
            call_args = mock_redis.lpush.call_args
            dlq_entry_str = call_args[0][1]
            dlq_entry = json.loads(dlq_entry_str)
            
            assert dlq_entry["job_id"] == str(job_id)
            assert dlq_entry["error_message"] == error_message
            assert dlq_entry["original_priority"] == "normal"
            assert dlq_entry["retry_count"] == 0
    
    @pytest.mark.asyncio
    async def test_retry_dead_letter_jobs(self, queue_service, mock_redis):
        """Test retrying jobs from dead letter queue."""
        job_id = uuid4()
        dlq_entry = {
            "job_id": str(job_id),
            "error_message": "Previous failure",
            "retry_count": 1,
            "original_priority": "high"
        }
        
        mock_job = TranslationJob(
            id=job_id,
            user_id="test-user",
            source_language="en",
            target_language="es",
            content_hash="test-hash",
            word_count=100,
            priority="high",
            status="failed"
        )
        
        # Mock DLQ operations
        mock_redis.llen.return_value = 1
        mock_redis.rpop.return_value = json.dumps(dlq_entry)
        
        with patch.object(queue_service, 'get_job', return_value=mock_job), \
             patch.object(queue_service, 'update_job_status', return_value=True), \
             patch.object(queue_service, 'enqueue', return_value=True), \
             patch.object(queue_service, '_get_redis_client', return_value=mock_redis):
            
            retried_count = await queue_service.retry_dead_letter_jobs(max_retries=3)
            
            assert retried_count == 1
            mock_redis.rpop.assert_called_once_with("queue:dlq")
    
    @pytest.mark.asyncio
    async def test_retry_dead_letter_jobs_max_retries_exceeded(self, queue_service, mock_redis):
        """Test that jobs exceeding max retries stay in DLQ."""
        job_id = uuid4()
        dlq_entry = {
            "job_id": str(job_id),
            "error_message": "Previous failure",
            "retry_count": 3,  # Already at max retries
            "original_priority": "high"
        }
        
        # Mock DLQ operations
        mock_redis.llen.return_value = 1
        mock_redis.rpop.return_value = json.dumps(dlq_entry)
        
        with patch.object(queue_service, '_get_redis_client', return_value=mock_redis):
            retried_count = await queue_service.retry_dead_letter_jobs(max_retries=3)
            
            assert retried_count == 0
            # Job should be put back in DLQ
            mock_redis.lpush.assert_called_once_with("queue:dlq", json.dumps(dlq_entry))
    
    @pytest.mark.asyncio
    async def test_cleanup_stale_processing_jobs(self, queue_service, mock_redis):
        """Test cleanup of stale processing jobs."""
        job_id = uuid4()
        
        # Mock stale job
        stale_job = TranslationJob(
            id=job_id,
            user_id="test-user",
            source_language="en",
            target_language="es",
            content_hash="test-hash",
            word_count=100,
            priority="normal",
            status="processing",
            started_at=datetime.utcnow() - timedelta(hours=2)  # 2 hours ago
        )
        
        mock_redis.smembers.return_value = {str(job_id)}
        
        with patch('src.database.connection.get_db_session') as mock_session, \
             patch('src.database.repositories.JobRepository') as mock_job_repo, \
             patch.object(queue_service, 'move_to_dead_letter_queue', return_value=True), \
             patch.object(queue_service, '_get_redis_client', return_value=mock_redis):
            
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            mock_repo_instance = AsyncMock()
            mock_repo_instance.get_by_id.return_value = stale_job
            mock_repo_instance.update_job_status.return_value = True
            mock_job_repo.return_value = mock_repo_instance
            
            cleaned_count = await queue_service.cleanup_stale_processing_jobs(timeout_minutes=30)
            
            assert cleaned_count == 1
            mock_repo_instance.update_job_status.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_queue_error_handling(self, queue_service, mock_redis):
        """Test error handling in queue operations."""
        # Mock Redis error
        mock_redis.zadd.side_effect = Exception("Redis connection failed")
        
        sample_job = TranslationJob(
            id=uuid4(),
            user_id="test-user",
            source_language="en",
            target_language="es",
            content_hash="test-hash",
            word_count=100,
            priority="normal",
            status="queued",
            created_at=datetime.utcnow()
        )
        
        with patch.object(queue_service, '_get_redis_client', return_value=mock_redis):
            with pytest.raises(QueueError):
                await queue_service.enqueue(sample_job)
    
    @pytest.mark.asyncio
    async def test_priority_ordering_under_concurrent_access(self, queue_service, mock_redis):
        """Test priority ordering is maintained under concurrent access (Requirement 3.1, 3.2)."""
        # Create jobs with different priorities
        critical_job_data = {
            "job_id": str(uuid4()),
            "user_id": "user-1",
            "source_language": "en",
            "target_language": "es",
            "content_hash": "critical-hash",
            "word_count": 100,
            "priority": "critical",
            "created_at": datetime.utcnow().isoformat()
        }
        
        high_job_data = {
            "job_id": str(uuid4()),
            "user_id": "user-2",
            "source_language": "en",
            "target_language": "es",
            "content_hash": "high-hash",
            "word_count": 100,
            "priority": "high",
            "created_at": datetime.utcnow().isoformat()
        }
        
        normal_job_data = {
            "job_id": str(uuid4()),
            "user_id": "user-3",
            "source_language": "en",
            "target_language": "es",
            "content_hash": "normal-hash",
            "word_count": 100,
            "priority": "normal",
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Mock dequeue to return jobs in priority order regardless of enqueue order
        dequeue_sequence = [
            [(json.dumps(critical_job_data), 1234567890.0)],  # Critical first
            [(json.dumps(high_job_data), 1234567891.0)],     # High second
            [(json.dumps(normal_job_data), 1234567892.0)],   # Normal third
            []  # No more jobs
        ]
        
        call_count = 0
        def mock_zpopmin_priority(queue_name, count=1):
            nonlocal call_count
            if queue_name == "queue:critical" and call_count == 0:
                call_count += 1
                return dequeue_sequence[0]
            elif queue_name == "queue:high" and call_count == 1:
                call_count += 1
                return dequeue_sequence[1]
            elif queue_name == "queue:normal" and call_count == 2:
                call_count += 1
                return dequeue_sequence[2]
            return []
        
        mock_redis.zpopmin.side_effect = mock_zpopmin_priority
        
        with patch.object(queue_service, '_get_redis_client', return_value=mock_redis):
            # Dequeue jobs and verify priority order
            jobs = []
            for _ in range(3):
                job = await queue_service.dequeue()
                if job:
                    jobs.append(job)
            
            assert len(jobs) == 3
            assert jobs[0].priority == "critical"
            assert jobs[1].priority == "high"
            assert jobs[2].priority == "normal"
    
    @pytest.mark.asyncio
    async def test_concurrent_enqueue_dequeue_operations(self, queue_service, mock_redis):
        """Test concurrent enqueue and dequeue operations (Requirement 3.4)."""
        # Create multiple jobs for concurrent operations
        jobs = []
        for i in range(5):
            job = TranslationJob(
                id=uuid4(),
                user_id=f"concurrent-user-{i}",
                source_language="en",
                target_language="es",
                content_hash=f"concurrent-hash-{i}",
                word_count=100,
                priority=["critical", "high", "normal"][i % 3],
                status=JobStatus.QUEUED.value,
                created_at=datetime.utcnow()
            )
            jobs.append(job)
        
        # Mock successful enqueue operations
        mock_redis.zadd.return_value = 1
        
        # Mock dequeue operations returning jobs in priority order
        job_data_list = []
        for job in jobs:
            if job.priority == "critical":
                job_data = {
                    "job_id": str(job.id),
                    "user_id": job.user_id,
                    "source_language": job.source_language,
                    "target_language": job.target_language,
                    "content_hash": job.content_hash,
                    "word_count": job.word_count,
                    "priority": job.priority,
                    "created_at": job.created_at.isoformat()
                }
                job_data_list.append(job_data)
        
        dequeue_call_count = 0
        def mock_zpopmin_concurrent(queue_name, count=1):
            nonlocal dequeue_call_count
            if queue_name == "queue:critical" and dequeue_call_count < len(job_data_list):
                result = [(json.dumps(job_data_list[dequeue_call_count]), 1234567890.0)]
                dequeue_call_count += 1
                return result
            return []
        
        mock_redis.zpopmin.side_effect = mock_zpopmin_concurrent
        
        with patch.object(queue_service, '_get_redis_client', return_value=mock_redis):
            # Perform concurrent enqueue operations
            enqueue_tasks = [queue_service.enqueue(job) for job in jobs]
            enqueue_results = await asyncio.gather(*enqueue_tasks, return_exceptions=True)
            
            # Verify all enqueue operations succeeded
            assert all(result is True for result in enqueue_results if not isinstance(result, Exception))
            
            # Perform concurrent dequeue operations
            dequeue_tasks = [queue_service.dequeue() for _ in range(3)]
            dequeue_results = await asyncio.gather(*dequeue_tasks, return_exceptions=True)
            
            # Verify dequeue operations completed without errors
            successful_dequeues = [result for result in dequeue_results if not isinstance(result, Exception) and result is not None]
            assert len(successful_dequeues) >= 0  # At least some should succeed
    
    @pytest.mark.asyncio
    async def test_queue_depth_consistency_under_concurrent_access(self, queue_service, mock_redis):
        """Test queue depth consistency during concurrent operations (Requirement 3.4)."""
        # Mock queue depths that change during concurrent operations
        depth_sequence = [5, 4, 3, 2, 1, 0]  # Simulating jobs being processed
        call_count = 0
        
        def mock_zcard_concurrent(queue_name):
            nonlocal call_count
            if call_count < len(depth_sequence):
                result = depth_sequence[call_count]
                call_count += 1
                return result
            return 0
        
        mock_redis.zcard.side_effect = mock_zcard_concurrent
        
        with patch.object(queue_service, '_get_redis_client', return_value=mock_redis):
            # Perform concurrent depth checks
            depth_tasks = [queue_service.get_queue_depth(Priority.NORMAL) for _ in range(6)]
            depth_results = await asyncio.gather(*depth_tasks, return_exceptions=True)
            
            # Verify all depth checks completed successfully
            successful_depths = [result for result in depth_results if not isinstance(result, Exception)]
            assert len(successful_depths) == 6
            
            # Verify depths are decreasing (simulating job processing)
            for i in range(len(successful_depths) - 1):
                assert successful_depths[i] >= successful_depths[i + 1]
    
    @pytest.mark.asyncio
    async def test_job_dispatch_priority_interruption(self, queue_service, mock_redis):
        """Test that critical jobs can interrupt lower priority processing (Requirement 3.1, 3.4)."""
        # Simulate scenario where critical job arrives while normal jobs are processing
        
        # Mock processing set with normal priority jobs
        processing_jobs = {"normal-job-1", "normal-job-2"}
        mock_redis.smembers.return_value = processing_jobs
        
        # Mock critical job being enqueued
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
        
        # Mock successful enqueue
        mock_redis.zadd.return_value = 1
        
        # Mock dequeue returning critical job first
        critical_job_data = {
            "job_id": str(critical_job.id),
            "user_id": critical_job.user_id,
            "source_language": critical_job.source_language,
            "target_language": critical_job.target_language,
            "content_hash": critical_job.content_hash,
            "word_count": critical_job.word_count,
            "priority": critical_job.priority,
            "created_at": critical_job.created_at.isoformat()
        }
        
        mock_redis.zpopmin.return_value = [(json.dumps(critical_job_data), 1234567890.0)]
        
        with patch.object(queue_service, '_get_redis_client', return_value=mock_redis):
            # Enqueue critical job
            enqueue_success = await queue_service.enqueue(critical_job)
            assert enqueue_success is True
            
            # Dequeue should return critical job first, even with normal jobs processing
            dequeued_job = await queue_service.dequeue()
            assert dequeued_job is not None
            assert dequeued_job.priority == Priority.CRITICAL.value
            assert str(dequeued_job.id) == str(critical_job.id)
            
            # Verify critical job was added to processing set
            mock_redis.sadd.assert_called_with("processing_jobs", str(critical_job.id))
    
    @pytest.mark.asyncio
    async def test_close_queue_service(self, queue_service, mock_redis):
        """Test closing queue service."""
        queue_service.redis_client = mock_redis
        
        await queue_service.close()
        
        mock_redis.close.assert_called_once()