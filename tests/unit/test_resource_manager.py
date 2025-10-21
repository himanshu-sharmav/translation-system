"""
Unit tests for ResourceManager.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from src.database.models import ComputeInstance
from src.services.resource_manager import AutoScalingResourceManager
from src.utils.exceptions import ResourceScalingError


class TestAutoScalingResourceManager:
    """Test auto-scaling resource manager functionality."""
    
    @pytest.fixture
    def resource_manager(self):
        """Create resource manager instance."""
        return AutoScalingResourceManager()
    
    @pytest.fixture
    def sample_instance(self):
        """Create sample compute instance."""
        return ComputeInstance(
            id="test-instance-1",
            instance_type="g4dn.xlarge",
            status="running",
            gpu_utilization=50.0,
            memory_usage=60.0,
            active_jobs=2,
            max_concurrent_jobs=4,
            created_at=datetime.utcnow(),
            last_heartbeat=datetime.utcnow()
        )
    
    @pytest.mark.asyncio
    async def test_resource_manager_initialization(self, resource_manager):
        """Test resource manager initialization."""
        assert resource_manager.active_instances == {}
        assert resource_manager.scaling_in_progress == set()
        assert resource_manager.min_instances > 0
        assert resource_manager.max_instances > resource_manager.min_instances
        assert resource_manager.scale_up_threshold > 0
        assert resource_manager.scale_down_threshold < resource_manager.scale_up_threshold
    
    @pytest.mark.asyncio
    async def test_get_all_instances(self, resource_manager, sample_instance):
        """Test getting all instances."""
        # Initially empty
        instances = await resource_manager.get_all_instances()
        assert len(instances) == 0
        
        # Add instance
        resource_manager.active_instances[sample_instance.id] = sample_instance
        instances = await resource_manager.get_all_instances()
        assert len(instances) == 1
        assert instances[0].id == sample_instance.id
    
    @pytest.mark.asyncio
    async def test_health_check_healthy_instance(self, resource_manager, sample_instance):
        """Test health check for healthy instance."""
        resource_manager.active_instances[sample_instance.id] = sample_instance
        
        is_healthy = await resource_manager.health_check(sample_instance.id)
        assert is_healthy is True
    
    @pytest.mark.asyncio
    async def test_health_check_nonexistent_instance(self, resource_manager):
        """Test health check for nonexistent instance."""
        is_healthy = await resource_manager.health_check("nonexistent-instance")
        assert is_healthy is False
    
    @pytest.mark.asyncio
    async def test_health_check_stale_heartbeat(self, resource_manager, sample_instance):
        """Test health check for instance with stale heartbeat."""
        # Set old heartbeat
        sample_instance.last_heartbeat = datetime.utcnow() - timedelta(minutes=10)
        resource_manager.active_instances[sample_instance.id] = sample_instance
        
        is_healthy = await resource_manager.health_check(sample_instance.id)
        assert is_healthy is False
    
    @pytest.mark.asyncio
    async def test_health_check_stopped_instance(self, resource_manager, sample_instance):
        """Test health check for stopped instance."""
        sample_instance.status = "stopped"
        resource_manager.active_instances[sample_instance.id] = sample_instance
        
        is_healthy = await resource_manager.health_check(sample_instance.id)
        assert is_healthy is False
    
    @pytest.mark.asyncio
    async def test_get_instance_metrics(self, resource_manager, sample_instance):
        """Test getting instance metrics."""
        resource_manager.active_instances[sample_instance.id] = sample_instance
        
        with patch('src.database.connection.get_db_session') as mock_session, \
             patch('src.database.repositories.MetricsRepository') as mock_metrics_repo:
            
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            mock_repo_instance = AsyncMock()
            mock_repo_instance.get_metrics.return_value = []  # No historical metrics
            mock_metrics_repo.return_value = mock_repo_instance
            
            metrics = await resource_manager.get_instance_metrics(sample_instance.id)
            
            assert isinstance(metrics, dict)
            assert "cpu_utilization" in metrics
            assert "gpu_utilization" in metrics
            assert "memory_usage" in metrics
            assert "active_jobs" in metrics
            assert "max_jobs" in metrics
            assert "utilization_ratio" in metrics
            
            # Check utilization ratio calculation
            expected_ratio = sample_instance.active_jobs / sample_instance.max_concurrent_jobs
            assert metrics["utilization_ratio"] == expected_ratio
    
    @pytest.mark.asyncio
    async def test_get_instance_metrics_nonexistent(self, resource_manager):
        """Test getting metrics for nonexistent instance."""
        metrics = await resource_manager.get_instance_metrics("nonexistent")
        assert metrics == {}
    
    @pytest.mark.asyncio
    async def test_calculate_scaling_recommendation_scale_up(self, resource_manager):
        """Test scaling recommendation for scale up scenarios."""
        # High queue depth and utilization
        recommendation = await resource_manager._calculate_scaling_recommendation(
            queue_depth=20,
            utilization_ratio=0.9,
            healthy_instances=2
        )
        assert recommendation == "scale_up"
        
        # High queue depth regardless of utilization
        recommendation = await resource_manager._calculate_scaling_recommendation(
            queue_depth=16,
            utilization_ratio=0.3,
            healthy_instances=3
        )
        assert recommendation == "scale_up"
        
        # Below minimum instances
        resource_manager.min_instances = 3
        recommendation = await resource_manager._calculate_scaling_recommendation(
            queue_depth=5,
            utilization_ratio=0.5,
            healthy_instances=2
        )
        assert recommendation == "scale_up"
    
    @pytest.mark.asyncio
    async def test_calculate_scaling_recommendation_scale_down(self, resource_manager):
        """Test scaling recommendation for scale down scenarios."""
        resource_manager.min_instances = 1
        
        # Low queue and utilization, above minimum instances
        recommendation = await resource_manager._calculate_scaling_recommendation(
            queue_depth=0,
            utilization_ratio=0.1,
            healthy_instances=3
        )
        assert recommendation == "scale_down"
    
    @pytest.mark.asyncio
    async def test_calculate_scaling_recommendation_maintain(self, resource_manager):
        """Test scaling recommendation for maintain scenarios."""
        # Moderate queue and utilization
        recommendation = await resource_manager._calculate_scaling_recommendation(
            queue_depth=3,
            utilization_ratio=0.5,
            healthy_instances=2
        )
        assert recommendation == "maintain"
    
    @pytest.mark.asyncio
    async def test_can_scale_timing(self, resource_manager):
        """Test scaling timing restrictions."""
        # Initially should be able to scale
        assert resource_manager._can_scale() is True
        
        # After setting recent scale action, should not be able to scale
        resource_manager.last_scale_action = datetime.utcnow()
        assert resource_manager._can_scale() is False
        
        # After enough time has passed, should be able to scale again
        resource_manager.last_scale_action = datetime.utcnow() - timedelta(minutes=10)
        assert resource_manager._can_scale() is True
    
    @pytest.mark.asyncio
    async def test_select_instance_type(self, resource_manager):
        """Test instance type selection logic."""
        # With no existing instances, should prefer GPU
        instance_type = await resource_manager._select_instance_type()
        assert "gpu" in instance_type.lower() or "g4" in instance_type
        
        # With many GPU instances, should prefer CPU for balance
        for i in range(5):
            gpu_instance = ComputeInstance(
                id=f"gpu-instance-{i}",
                instance_type="g4dn.xlarge",
                status="running",
                gpu_utilization=50.0,
                memory_usage=60.0,
                active_jobs=1,
                max_concurrent_jobs=4,
                created_at=datetime.utcnow(),
                last_heartbeat=datetime.utcnow()
            )
            resource_manager.active_instances[gpu_instance.id] = gpu_instance
        
        instance_type = await resource_manager._select_instance_type()
        # Should now prefer CPU for balance
        assert "cpu" in instance_type.lower() or "r5" in instance_type
    
    @pytest.mark.asyncio
    async def test_get_max_jobs_for_instance_type(self, resource_manager):
        """Test max jobs calculation for different instance types."""
        # GPU instances should handle more jobs
        gpu_max = resource_manager._get_max_jobs_for_instance_type("g4dn.xlarge")
        assert gpu_max == 4
        
        p3_max = resource_manager._get_max_jobs_for_instance_type("p3.2xlarge")
        assert p3_max == 4
        
        # CPU instances should handle fewer jobs
        cpu_max = resource_manager._get_max_jobs_for_instance_type("r5.8xlarge")
        assert cpu_max == 2
    
    @pytest.mark.asyncio
    async def test_select_instances_for_removal(self, resource_manager):
        """Test instance selection for removal."""
        # Create instances with different characteristics
        idle_instance = ComputeInstance(
            id="idle-instance",
            instance_type="g4dn.xlarge",
            status="running",
            gpu_utilization=10.0,
            memory_usage=20.0,
            active_jobs=0,  # No active jobs
            max_concurrent_jobs=4,
            created_at=datetime.utcnow() - timedelta(hours=2),  # Old instance
            last_heartbeat=datetime.utcnow() - timedelta(minutes=30)  # Idle
        )
        
        busy_instance = ComputeInstance(
            id="busy-instance",
            instance_type="g4dn.xlarge",
            status="running",
            gpu_utilization=80.0,
            memory_usage=90.0,
            active_jobs=3,  # Has active jobs
            max_concurrent_jobs=4,
            created_at=datetime.utcnow(),
            last_heartbeat=datetime.utcnow()
        )
        
        recent_idle_instance = ComputeInstance(
            id="recent-idle-instance",
            instance_type="g4dn.xlarge",
            status="running",
            gpu_utilization=5.0,
            memory_usage=15.0,
            active_jobs=0,  # No active jobs
            max_concurrent_jobs=4,
            created_at=datetime.utcnow(),  # Recent instance
            last_heartbeat=datetime.utcnow()
        )
        
        resource_manager.active_instances = {
            idle_instance.id: idle_instance,
            busy_instance.id: busy_instance,
            recent_idle_instance.id: recent_idle_instance
        }
        
        # Select 2 instances for removal
        selected = await resource_manager._select_instances_for_removal(2)
        
        # Should select idle instances, preferring the one that's been idle longer
        assert len(selected) == 2
        selected_ids = [inst.id for inst in selected]
        
        # Should not select the busy instance
        assert busy_instance.id not in selected_ids
        
        # Should prefer the instance that's been idle longer
        assert idle_instance.id in selected_ids
        assert recent_idle_instance.id in selected_ids
    
    @pytest.mark.asyncio
    async def test_scale_up_success(self, resource_manager):
        """Test successful scale up operation."""
        with patch.object(resource_manager, '_can_scale', return_value=True), \
             patch.object(resource_manager, '_create_compute_instance', return_value="new-instance-1"), \
             patch.object(resource_manager, '_wait_for_instances_ready'):
            
            # Set current instances below max
            resource_manager.max_instances = 5
            resource_manager.active_instances = {}
            
            result = await resource_manager.scale_up(2)
            
            assert len(result) == 2
            assert "new-instance-1" in result
    
    @pytest.mark.asyncio
    async def test_scale_up_at_max_capacity(self, resource_manager):
        """Test scale up when at maximum capacity."""
        with patch.object(resource_manager, '_can_scale', return_value=True):
            # Set at max capacity
            resource_manager.max_instances = 2
            resource_manager.active_instances = {
                "instance-1": MagicMock(),
                "instance-2": MagicMock()
            }
            
            result = await resource_manager.scale_up(1)
            
            assert len(result) == 0  # Should not scale up
    
    @pytest.mark.asyncio
    async def test_scale_up_blocked_by_timing(self, resource_manager):
        """Test scale up blocked by recent scaling activity."""
        with patch.object(resource_manager, '_can_scale', return_value=False):
            result = await resource_manager.scale_up(1)
            assert len(result) == 0
    
    @pytest.mark.asyncio
    async def test_scale_down_success(self, resource_manager):
        """Test successful scale down operation."""
        # Create instances for removal
        instance1 = ComputeInstance(
            id="instance-1",
            instance_type="g4dn.xlarge",
            status="running",
            active_jobs=0,
            max_concurrent_jobs=4,
            created_at=datetime.utcnow(),
            last_heartbeat=datetime.utcnow()
        )
        
        instance2 = ComputeInstance(
            id="instance-2",
            instance_type="g4dn.xlarge",
            status="running",
            active_jobs=0,
            max_concurrent_jobs=4,
            created_at=datetime.utcnow(),
            last_heartbeat=datetime.utcnow()
        )
        
        resource_manager.active_instances = {
            instance1.id: instance1,
            instance2.id: instance2
        }
        resource_manager.min_instances = 1
        
        with patch.object(resource_manager, '_can_scale', return_value=True), \
             patch.object(resource_manager, '_select_instances_for_removal', return_value=[instance1]), \
             patch.object(resource_manager, '_terminate_compute_instance', return_value=True):
            
            result = await resource_manager.scale_down(1)
            
            assert len(result) == 1
            assert instance1.id in result
            assert instance1.id not in resource_manager.active_instances
    
    @pytest.mark.asyncio
    async def test_scale_down_at_min_capacity(self, resource_manager):
        """Test scale down when at minimum capacity."""
        with patch.object(resource_manager, '_can_scale', return_value=True):
            # Set at min capacity
            resource_manager.min_instances = 2
            resource_manager.active_instances = {
                "instance-1": MagicMock(),
                "instance-2": MagicMock()
            }
            
            result = await resource_manager.scale_down(1)
            
            assert len(result) == 0  # Should not scale down
    
    @pytest.mark.asyncio
    async def test_get_scaling_metrics(self, resource_manager):
        """Test getting comprehensive scaling metrics."""
        # Add some instances
        instance = ComputeInstance(
            id="test-instance",
            instance_type="g4dn.xlarge",
            status="running",
            gpu_utilization=50.0,
            memory_usage=60.0,
            active_jobs=2,
            max_concurrent_jobs=4,
            created_at=datetime.utcnow(),
            last_heartbeat=datetime.utcnow()
        )
        resource_manager.active_instances[instance.id] = instance
        
        with patch.object(resource_manager.queue_service, 'get_queue_stats') as mock_queue_stats, \
             patch.object(resource_manager, 'health_check', return_value=True), \
             patch.object(resource_manager, '_calculate_scaling_recommendation', return_value="maintain"):
            
            mock_queue_stats.return_value = {
                "critical": 2,
                "high": 5,
                "normal": 10,
                "processing": 3
            }
            
            metrics = await resource_manager.get_scaling_metrics()
            
            assert "timestamp" in metrics
            assert "instances" in metrics
            assert "queue" in metrics
            assert "utilization" in metrics
            assert "scaling" in metrics
            
            # Check instance metrics
            assert metrics["instances"]["total"] == 1
            assert metrics["instances"]["healthy"] == 1
            
            # Check queue metrics
            assert metrics["queue"]["total_queued"] == 17  # 2+5+10
            assert metrics["queue"]["critical"] == 2
            
            # Check utilization metrics
            assert metrics["utilization"]["total_capacity"] == 4
            assert metrics["utilization"]["active_jobs"] == 2
            assert metrics["utilization"]["utilization_ratio"] == 0.5
            
            # Check scaling metrics
            assert metrics["scaling"]["recommendation"] == "maintain"
    
    @pytest.mark.asyncio
    async def test_monitoring_loop_error_handling(self, resource_manager):
        """Test monitoring loop handles errors gracefully."""
        with patch.object(resource_manager, '_perform_scaling_check', side_effect=Exception("Test error")), \
             patch.object(resource_manager, '_cleanup_unhealthy_instances'), \
             patch.object(resource_manager, '_update_instance_heartbeats'):
            
            resource_manager._running = True
            
            # Start monitoring loop
            task = asyncio.create_task(resource_manager._monitoring_loop())
            
            # Let it run briefly
            await asyncio.sleep(0.1)
            
            # Stop monitoring
            resource_manager._running = False
            
            # Wait for task to complete
            try:
                await asyncio.wait_for(task, timeout=1.0)
            except asyncio.TimeoutError:
                task.cancel()
            
            # Should not raise exception despite error in _perform_scaling_check
    
    @pytest.mark.asyncio
    async def test_cleanup_unhealthy_instances(self, resource_manager):
        """Test cleanup of unhealthy instances."""
        # Add healthy and unhealthy instances
        healthy_instance = ComputeInstance(
            id="healthy-instance",
            instance_type="g4dn.xlarge",
            status="running",
            active_jobs=1,
            max_concurrent_jobs=4,
            created_at=datetime.utcnow(),
            last_heartbeat=datetime.utcnow()
        )
        
        unhealthy_instance = ComputeInstance(
            id="unhealthy-instance",
            instance_type="g4dn.xlarge",
            status="running",
            active_jobs=0,
            max_concurrent_jobs=4,
            created_at=datetime.utcnow(),
            last_heartbeat=datetime.utcnow() - timedelta(minutes=10)  # Stale heartbeat
        )
        
        resource_manager.active_instances = {
            healthy_instance.id: healthy_instance,
            unhealthy_instance.id: unhealthy_instance
        }
        
        with patch('src.database.connection.get_db_session') as mock_session, \
             patch('src.database.repositories.base.BaseRepository') as mock_repo:
            
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            mock_repo_instance = AsyncMock()
            mock_repo_instance.get_by_id.return_value = unhealthy_instance
            mock_repo_instance.update.return_value = True
            mock_repo.return_value = mock_repo_instance
            
            await resource_manager._cleanup_unhealthy_instances()
            
            # Healthy instance should remain
            assert healthy_instance.id in resource_manager.active_instances
            
            # Unhealthy instance should be removed
            assert unhealthy_instance.id not in resource_manager.active_instances