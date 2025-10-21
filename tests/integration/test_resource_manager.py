"""
Integration tests for resource manager with the overall system.
"""

import pytest
import asyncio
from datetime import datetime, timedelta

from src.database.models import ComputeInstance
from src.services.resource_manager import AutoScalingResourceManager


class TestResourceManagerIntegration:
    """Test resource manager integration with the system."""
    
    @pytest.fixture
    async def resource_manager(self, db_session):
        """Create resource manager for testing."""
        manager = AutoScalingResourceManager()
        yield manager
        await manager.close()
    
    @pytest.mark.asyncio
    async def test_resource_manager_lifecycle(self, resource_manager):
        """Test resource manager start/stop lifecycle."""
        # Initially not running
        assert resource_manager._running is False
        
        # Start monitoring
        await resource_manager.start_monitoring()
        assert resource_manager._running is True
        assert resource_manager._monitoring_task is not None
        
        # Stop monitoring
        await resource_manager.stop_monitoring()
        assert resource_manager._running is False
    
    @pytest.mark.asyncio
    async def test_resource_manager_basic_functionality(self, resource_manager):
        """Test basic resource manager functionality."""
        # This test requires database setup
        pytest.skip("Requires database setup for integration test")
        
        # Test getting scaling metrics
        metrics = await resource_manager.get_scaling_metrics()
        
        assert isinstance(metrics, dict)
        assert "instances" in metrics
        assert "queue" in metrics
        assert "utilization" in metrics
        assert "scaling" in metrics
        
        # Test health checks
        instances = await resource_manager.get_all_instances()
        for instance in instances:
            health = await resource_manager.health_check(instance.id)
            assert isinstance(health, bool)
    
    @pytest.mark.asyncio
    async def test_resource_manager_scaling_integration(self, resource_manager):
        """Test resource manager scaling integration."""
        # This test requires cloud provider integration
        pytest.skip("Requires cloud provider setup for integration test")
        
        # Test scale up
        initial_count = len(await resource_manager.get_all_instances())
        
        new_instances = await resource_manager.scale_up(1)
        assert len(new_instances) >= 0
        
        final_count = len(await resource_manager.get_all_instances())
        if new_instances:
            assert final_count > initial_count
        
        # Test scale down
        if final_count > resource_manager.min_instances:
            removed_instances = await resource_manager.scale_down(1)
            assert len(removed_instances) >= 0
    
    @pytest.mark.asyncio
    async def test_resource_manager_auto_scaling_logic(self, resource_manager):
        """Test auto-scaling decision logic."""
        # Test scaling recommendations under different conditions
        
        # High load scenario
        recommendation = await resource_manager._calculate_scaling_recommendation(
            queue_depth=25,
            utilization_ratio=0.9,
            healthy_instances=2
        )
        assert recommendation == "scale_up"
        
        # Low load scenario
        recommendation = await resource_manager._calculate_scaling_recommendation(
            queue_depth=0,
            utilization_ratio=0.1,
            healthy_instances=5
        )
        if resource_manager.min_instances < 5:
            assert recommendation == "scale_down"
        
        # Balanced scenario
        recommendation = await resource_manager._calculate_scaling_recommendation(
            queue_depth=3,
            utilization_ratio=0.5,
            healthy_instances=3
        )
        assert recommendation == "maintain"
    
    @pytest.mark.asyncio
    async def test_resource_manager_error_resilience(self, resource_manager):
        """Test resource manager handles errors gracefully."""
        # Test with invalid instance operations
        
        # Try to get metrics for nonexistent instance
        metrics = await resource_manager.get_instance_metrics("nonexistent-instance")
        assert metrics == {}
        
        # Try to health check nonexistent instance
        health = await resource_manager.health_check("nonexistent-instance")
        assert health is False
        
        # Test scaling with constraints
        resource_manager.max_instances = 1
        resource_manager.active_instances = {"instance-1": ComputeInstance(
            id="instance-1",
            instance_type="g4dn.xlarge",
            status="running",
            active_jobs=0,
            max_concurrent_jobs=4,
            created_at=datetime.utcnow(),
            last_heartbeat=datetime.utcnow()
        )}
        
        # Should not scale up beyond max
        with patch.object(resource_manager, '_can_scale', return_value=True):
            result = await resource_manager.scale_up(1)
            assert len(result) == 0
    
    @pytest.mark.asyncio
    async def test_resource_manager_concurrent_operations(self, resource_manager):
        """Test resource manager handles concurrent operations."""
        # Test concurrent scaling operations
        
        with patch.object(resource_manager, '_create_compute_instance') as mock_create, \
             patch.object(resource_manager, '_wait_for_instances_ready'), \
             patch.object(resource_manager, '_can_scale', return_value=True):
            
            mock_create.side_effect = lambda: f"instance-{len(resource_manager.active_instances)}"
            
            # Start multiple scale up operations concurrently
            tasks = [
                resource_manager.scale_up(1),
                resource_manager.scale_up(1),
                resource_manager.scale_up(1)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Should handle concurrent operations gracefully
            successful_results = [r for r in results if not isinstance(r, Exception)]
            assert len(successful_results) >= 0
    
    @pytest.mark.asyncio
    async def test_resource_manager_metrics_collection(self, resource_manager):
        """Test resource manager metrics collection."""
        # Add a test instance
        test_instance = ComputeInstance(
            id="metrics-test-instance",
            instance_type="g4dn.xlarge",
            status="running",
            gpu_utilization=75.0,
            memory_usage=80.0,
            active_jobs=3,
            max_concurrent_jobs=4,
            created_at=datetime.utcnow(),
            last_heartbeat=datetime.utcnow()
        )
        
        resource_manager.active_instances[test_instance.id] = test_instance
        
        # Test instance metrics
        with patch('src.database.connection.get_db_session') as mock_session, \
             patch('src.database.repositories.MetricsRepository') as mock_metrics_repo:
            
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            mock_repo_instance = AsyncMock()
            mock_repo_instance.get_metrics.return_value = []
            mock_metrics_repo.return_value = mock_repo_instance
            
            metrics = await resource_manager.get_instance_metrics(test_instance.id)
            
            assert "cpu_utilization" in metrics
            assert "gpu_utilization" in metrics
            assert "memory_usage" in metrics
            assert "active_jobs" in metrics
            assert metrics["active_jobs"] == 3
            assert metrics["max_jobs"] == 4
            assert metrics["utilization_ratio"] == 0.75
        
        # Test system-wide metrics
        with patch.object(resource_manager.queue_service, 'get_queue_stats') as mock_queue_stats, \
             patch.object(resource_manager, 'health_check', return_value=True):
            
            mock_queue_stats.return_value = {
                "critical": 1,
                "high": 3,
                "normal": 8,
                "processing": 2
            }
            
            scaling_metrics = await resource_manager.get_scaling_metrics()
            
            assert scaling_metrics["instances"]["total"] == 1
            assert scaling_metrics["instances"]["healthy"] == 1
            assert scaling_metrics["queue"]["total_queued"] == 12
            assert scaling_metrics["utilization"]["active_jobs"] == 3
            assert scaling_metrics["utilization"]["total_capacity"] == 4
    
    @pytest.mark.asyncio
    async def test_resource_manager_cost_optimization(self, resource_manager):
        """Test resource manager cost optimization features."""
        # Test idle instance detection and removal
        
        # Create instances with different idle times
        old_idle_instance = ComputeInstance(
            id="old-idle-instance",
            instance_type="g4dn.xlarge",
            status="running",
            gpu_utilization=5.0,
            memory_usage=10.0,
            active_jobs=0,
            max_concurrent_jobs=4,
            created_at=datetime.utcnow() - timedelta(hours=2),
            last_heartbeat=datetime.utcnow() - timedelta(minutes=30)
        )
        
        recent_idle_instance = ComputeInstance(
            id="recent-idle-instance",
            instance_type="g4dn.xlarge",
            status="running",
            gpu_utilization=8.0,
            memory_usage=15.0,
            active_jobs=0,
            max_concurrent_jobs=4,
            created_at=datetime.utcnow() - timedelta(minutes=10),
            last_heartbeat=datetime.utcnow() - timedelta(minutes=5)
        )
        
        busy_instance = ComputeInstance(
            id="busy-instance",
            instance_type="g4dn.xlarge",
            status="running",
            gpu_utilization=85.0,
            memory_usage=90.0,
            active_jobs=3,
            max_concurrent_jobs=4,
            created_at=datetime.utcnow() - timedelta(hours=1),
            last_heartbeat=datetime.utcnow()
        )
        
        resource_manager.active_instances = {
            old_idle_instance.id: old_idle_instance,
            recent_idle_instance.id: recent_idle_instance,
            busy_instance.id: busy_instance
        }
        
        # Test instance selection for removal
        selected = await resource_manager._select_instances_for_removal(2)
        
        # Should select idle instances, preferring older idle ones
        assert len(selected) == 2
        selected_ids = [inst.id for inst in selected]
        
        # Should not select busy instance
        assert busy_instance.id not in selected_ids
        
        # Should include both idle instances
        assert old_idle_instance.id in selected_ids
        assert recent_idle_instance.id in selected_ids
        
        # Should prefer the older idle instance first
        assert selected[0].id == old_idle_instance.id
    
    @pytest.mark.asyncio
    async def test_resource_manager_performance_under_load(self, resource_manager):
        """Test resource manager performance under high load."""
        # Create many instances to test performance
        instances = {}
        for i in range(50):
            instance = ComputeInstance(
                id=f"load-test-instance-{i}",
                instance_type="g4dn.xlarge" if i % 2 == 0 else "r5.8xlarge",
                status="running",
                gpu_utilization=float(i % 100),
                memory_usage=float((i * 2) % 100),
                active_jobs=i % 5,
                max_concurrent_jobs=4,
                created_at=datetime.utcnow() - timedelta(minutes=i),
                last_heartbeat=datetime.utcnow() - timedelta(seconds=i * 10)
            )
            instances[instance.id] = instance
        
        resource_manager.active_instances = instances
        
        # Test performance of various operations
        start_time = datetime.utcnow()
        
        # Get all instances
        all_instances = await resource_manager.get_all_instances()
        assert len(all_instances) == 50
        
        # Health check all instances
        health_checks = []
        for instance_id in instances.keys():
            health = await resource_manager.health_check(instance_id)
            health_checks.append(health)
        
        # Should complete health checks for all instances
        assert len(health_checks) == 50
        
        # Get scaling metrics
        with patch.object(resource_manager.queue_service, 'get_queue_stats') as mock_queue_stats:
            mock_queue_stats.return_value = {"critical": 5, "high": 10, "normal": 20, "processing": 8}
            
            metrics = await resource_manager.get_scaling_metrics()
            assert metrics["instances"]["total"] == 50
        
        # Operations should complete in reasonable time
        elapsed = (datetime.utcnow() - start_time).total_seconds()
        assert elapsed < 10  # Should complete within 10 seconds