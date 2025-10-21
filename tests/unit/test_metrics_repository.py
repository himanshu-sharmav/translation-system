"""
Unit tests for MetricsRepository.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import uuid4

from src.database.models import SystemMetric
from tests.conftest import assert_metric_equals


class TestMetricsRepository:
    """Test MetricsRepository operations."""
    
    @pytest.mark.asyncio
    async def test_create_metric(self, metrics_repository, sample_system_metric):
        """Test creating a system metric."""
        created_metric = await metrics_repository.create(sample_system_metric)
        
        assert created_metric.metric_name == sample_system_metric.metric_name
        assert created_metric.metric_value == sample_system_metric.metric_value
        assert created_metric.instance_id == sample_system_metric.instance_id
        assert created_metric.timestamp is not None
    
    @pytest.mark.asyncio
    async def test_get_metric_by_id(self, metrics_repository, sample_system_metric):
        """Test getting a metric by ID."""
        # Create metric first
        await metrics_repository.create(sample_system_metric)
        
        # Retrieve metric
        retrieved_metric = await metrics_repository.get_by_id(sample_system_metric.id)
        
        assert retrieved_metric is not None
        assert_metric_equals(retrieved_metric, sample_system_metric)
    
    @pytest.mark.asyncio
    async def test_get_metrics_by_name_and_time_range(self, metrics_repository):
        """Test getting metrics by name within time range."""
        metric_name = "cpu_utilization"
        start_time = datetime.utcnow() - timedelta(hours=1)
        end_time = datetime.utcnow()
        
        # Create metrics within and outside the time range
        metrics_in_range = []
        for i in range(3):
            metric = SystemMetric(
                metric_name=metric_name,
                metric_value=Decimal(str(50 + i * 10)),
                instance_id=f"instance-{i}",
                timestamp=start_time + timedelta(minutes=i * 10)
            )
            metrics_in_range.append(metric)
            await metrics_repository.create(metric)
        
        # Create metric outside time range
        old_metric = SystemMetric(
            metric_name=metric_name,
            metric_value=Decimal('30.0'),
            instance_id="instance-old",
            timestamp=start_time - timedelta(hours=1)
        )
        await metrics_repository.create(old_metric)
        
        # Get metrics in range
        retrieved_metrics = await metrics_repository.get_metrics(
            metric_name, start_time, end_time
        )
        
        assert len(retrieved_metrics) == 3
        for metric in retrieved_metrics:
            assert metric.metric_name == metric_name
            assert start_time <= metric.timestamp <= end_time
    
    @pytest.mark.asyncio
    async def test_get_metrics_by_instance(self, metrics_repository):
        """Test getting metrics filtered by instance ID."""
        metric_name = "memory_usage"
        instance_id = "test-instance"
        start_time = datetime.utcnow() - timedelta(hours=1)
        end_time = datetime.utcnow()
        
        # Create metrics for specific instance
        instance_metric = SystemMetric(
            metric_name=metric_name,
            metric_value=Decimal('75.0'),
            instance_id=instance_id,
            timestamp=datetime.utcnow()
        )
        await metrics_repository.create(instance_metric)
        
        # Create metric for different instance
        other_metric = SystemMetric(
            metric_name=metric_name,
            metric_value=Decimal('60.0'),
            instance_id="other-instance",
            timestamp=datetime.utcnow()
        )
        await metrics_repository.create(other_metric)
        
        # Get metrics for specific instance
        instance_metrics = await metrics_repository.get_metrics(
            metric_name, start_time, end_time, instance_id
        )
        
        assert len(instance_metrics) == 1
        assert instance_metrics[0].instance_id == instance_id
    
    @pytest.mark.asyncio
    async def test_get_latest_metric(self, metrics_repository):
        """Test getting the latest metric value."""
        metric_name = "gpu_utilization"
        instance_id = "gpu-instance"
        
        # Create metrics with different timestamps
        old_metric = SystemMetric(
            metric_name=metric_name,
            metric_value=Decimal('60.0'),
            instance_id=instance_id,
            timestamp=datetime.utcnow() - timedelta(minutes=30)
        )
        
        latest_metric = SystemMetric(
            metric_name=metric_name,
            metric_value=Decimal('85.0'),
            instance_id=instance_id,
            timestamp=datetime.utcnow()
        )
        
        await metrics_repository.create(old_metric)
        await metrics_repository.create(latest_metric)
        
        # Get latest metric
        retrieved_latest = await metrics_repository.get_latest_metric(
            metric_name, instance_id
        )
        
        assert retrieved_latest is not None
        assert retrieved_latest.metric_value == Decimal('85.0')
        assert retrieved_latest.timestamp == latest_metric.timestamp
    
    @pytest.mark.asyncio
    async def test_get_latest_metric_without_instance(self, metrics_repository):
        """Test getting latest metric without instance filter."""
        metric_name = "system_load"
        
        # Create metrics for different instances
        metric1 = SystemMetric(
            metric_name=metric_name,
            metric_value=Decimal('1.5'),
            instance_id="instance-1",
            timestamp=datetime.utcnow() - timedelta(minutes=10)
        )
        
        metric2 = SystemMetric(
            metric_name=metric_name,
            metric_value=Decimal('2.0'),
            instance_id="instance-2",
            timestamp=datetime.utcnow()
        )
        
        await metrics_repository.create(metric1)
        await metrics_repository.create(metric2)
        
        # Get latest metric (should return the most recent regardless of instance)
        latest = await metrics_repository.get_latest_metric(metric_name)
        
        assert latest is not None
        assert latest.metric_value == Decimal('2.0')
    
    @pytest.mark.asyncio
    async def test_get_metric_aggregates(self, metrics_repository):
        """Test getting aggregated metrics."""
        metric_name = "response_time"
        start_time = datetime.utcnow() - timedelta(hours=1)
        end_time = datetime.utcnow()
        
        # Create metrics with different values
        values = [Decimal('10.0'), Decimal('20.0'), Decimal('30.0'), Decimal('40.0')]
        for i, value in enumerate(values):
            metric = SystemMetric(
                metric_name=metric_name,
                metric_value=value,
                instance_id=f"instance-{i}",
                timestamp=start_time + timedelta(minutes=i * 10)
            )
            await metrics_repository.create(metric)
        
        # Get aggregates
        aggregates = await metrics_repository.get_metric_aggregates(
            metric_name, start_time, end_time
        )
        
        assert aggregates["min"] == 10.0
        assert aggregates["max"] == 40.0
        assert aggregates["avg"] == 25.0  # (10+20+30+40)/4
        assert aggregates["sum"] == 100.0
        assert aggregates["count"] == 4
    
    @pytest.mark.asyncio
    async def test_get_metric_aggregates_with_instance_filter(self, metrics_repository):
        """Test getting aggregated metrics filtered by instance."""
        metric_name = "cpu_usage"
        instance_id = "target-instance"
        start_time = datetime.utcnow() - timedelta(hours=1)
        end_time = datetime.utcnow()
        
        # Create metrics for target instance
        target_values = [Decimal('50.0'), Decimal('60.0')]
        for i, value in enumerate(target_values):
            metric = SystemMetric(
                metric_name=metric_name,
                metric_value=value,
                instance_id=instance_id,
                timestamp=start_time + timedelta(minutes=i * 10)
            )
            await metrics_repository.create(metric)
        
        # Create metric for different instance
        other_metric = SystemMetric(
            metric_name=metric_name,
            metric_value=Decimal('90.0'),
            instance_id="other-instance",
            timestamp=start_time + timedelta(minutes=20)
        )
        await metrics_repository.create(other_metric)
        
        # Get aggregates for target instance only
        aggregates = await metrics_repository.get_metric_aggregates(
            metric_name, start_time, end_time, instance_id
        )
        
        assert aggregates["min"] == 50.0
        assert aggregates["max"] == 60.0
        assert aggregates["avg"] == 55.0
        assert aggregates["count"] == 2
    
    @pytest.mark.asyncio
    async def test_get_metrics_by_instance_all(self, metrics_repository):
        """Test getting all metrics for a specific instance."""
        instance_id = "test-instance"
        
        # Create metrics for the instance
        metrics = []
        metric_names = ["cpu_usage", "memory_usage", "disk_usage"]
        
        for metric_name in metric_names:
            metric = SystemMetric(
                metric_name=metric_name,
                metric_value=Decimal('75.0'),
                instance_id=instance_id,
                timestamp=datetime.utcnow()
            )
            metrics.append(metric)
            await metrics_repository.create(metric)
        
        # Create metric for different instance
        other_metric = SystemMetric(
            metric_name="cpu_usage",
            metric_value=Decimal('50.0'),
            instance_id="other-instance",
            timestamp=datetime.utcnow()
        )
        await metrics_repository.create(other_metric)
        
        # Get all metrics for the instance
        instance_metrics = await metrics_repository.get_metrics_by_instance(instance_id)
        
        assert len(instance_metrics) == 3
        for metric in instance_metrics:
            assert metric.instance_id == instance_id
    
    @pytest.mark.asyncio
    async def test_get_metric_names(self, metrics_repository):
        """Test getting all unique metric names."""
        # Create metrics with different names
        metric_names = ["cpu_usage", "memory_usage", "disk_usage", "cpu_usage"]  # cpu_usage repeated
        
        for i, name in enumerate(metric_names):
            metric = SystemMetric(
                metric_name=name,
                metric_value=Decimal('50.0'),
                instance_id=f"instance-{i}",
                timestamp=datetime.utcnow()
            )
            await metrics_repository.create(metric)
        
        # Get unique metric names
        names = await metrics_repository.get_metric_names()
        
        assert len(names) == 3  # Should be unique
        assert "cpu_usage" in names
        assert "memory_usage" in names
        assert "disk_usage" in names
    
    @pytest.mark.asyncio
    async def test_get_instance_ids(self, metrics_repository):
        """Test getting all unique instance IDs."""
        # Create metrics with different instance IDs
        instance_ids = ["instance-1", "instance-2", "instance-1", None]  # instance-1 repeated, None included
        
        for i, instance_id in enumerate(instance_ids):
            metric = SystemMetric(
                metric_name="test_metric",
                metric_value=Decimal('50.0'),
                instance_id=instance_id,
                timestamp=datetime.utcnow()
            )
            await metrics_repository.create(metric)
        
        # Get unique instance IDs (should exclude None)
        ids = await metrics_repository.get_instance_ids()
        
        assert len(ids) == 2  # Should be unique and exclude None
        assert "instance-1" in ids
        assert "instance-2" in ids
        assert None not in ids
    
    @pytest.mark.asyncio
    async def test_cleanup_old_metrics(self, metrics_repository):
        """Test cleaning up old metrics."""
        # Create old metrics
        old_time = datetime.utcnow() - timedelta(days=35)
        old_metrics = []
        for i in range(3):
            metric = SystemMetric(
                metric_name=f"old_metric_{i}",
                metric_value=Decimal('50.0'),
                instance_id=f"instance-{i}",
                timestamp=old_time
            )
            old_metrics.append(metric)
            await metrics_repository.create(metric)
        
        # Create recent metric
        recent_metric = SystemMetric(
            metric_name="recent_metric",
            metric_value=Decimal('75.0'),
            instance_id="recent-instance",
            timestamp=datetime.utcnow()
        )
        await metrics_repository.create(recent_metric)
        
        # Cleanup metrics older than 30 days
        deleted_count = await metrics_repository.cleanup_old_metrics(days=30)
        assert deleted_count == 3
        
        # Verify old metrics are deleted, recent metric remains
        remaining_metrics = await metrics_repository.get_all()
        assert len(remaining_metrics) == 1
        assert remaining_metrics[0].metric_name == "recent_metric"
    
    @pytest.mark.asyncio
    async def test_get_system_health_metrics(self, metrics_repository):
        """Test getting system health metrics."""
        # Create health-related metrics
        health_metrics = [
            ("gpu_utilization_percent", Decimal('80.0')),
            ("memory_usage_percent", Decimal('65.0')),
            ("cpu_utilization_percent", Decimal('45.0')),
            ("translation_latency_seconds", Decimal('2.5')),
            ("queue_depth_total", Decimal('10.0')),
            ("cache_hit_ratio", Decimal('0.85')),
            ("active_translation_jobs", Decimal('5.0'))
        ]
        
        for metric_name, value in health_metrics:
            # Create metrics for different instances
            for i in range(2):
                metric = SystemMetric(
                    metric_name=metric_name,
                    metric_value=value + i,
                    instance_id=f"instance-{i}",
                    timestamp=datetime.utcnow()
                )
                await metrics_repository.create(metric)
        
        # Get system health metrics
        health_data = await metrics_repository.get_system_health_metrics(minutes=5)
        
        assert "system_metrics" in health_data
        assert "instance_metrics" in health_data
        assert "timestamp" in health_data
        assert "time_window_minutes" in health_data
        
        # Check system metrics
        system_metrics = health_data["system_metrics"]
        assert "gpu_utilization_percent" in system_metrics
        assert "memory_usage_percent" in system_metrics
        
        # Check instance metrics
        instance_metrics = health_data["instance_metrics"]
        assert "instance-0" in instance_metrics
        assert "instance-1" in instance_metrics
    
    @pytest.mark.asyncio
    async def test_record_bulk_metrics(self, metrics_repository):
        """Test recording multiple metrics in bulk."""
        # Prepare bulk metrics data
        metrics_data = []
        for i in range(5):
            metric_data = {
                "metric_name": f"bulk_metric_{i}",
                "metric_value": 50.0 + i * 10,
                "instance_id": f"instance-{i}",
                "tags": {"batch": "test", "index": i}
            }
            metrics_data.append(metric_data)
        
        # Record bulk metrics
        success = await metrics_repository.record_bulk_metrics(metrics_data)
        assert success is True
        
        # Verify metrics were created
        all_metrics = await metrics_repository.get_all()
        bulk_metrics = [m for m in all_metrics if m.metric_name.startswith("bulk_metric_")]
        assert len(bulk_metrics) == 5
        
        for metric in bulk_metrics:
            assert metric.tags["batch"] == "test"
            assert "index" in metric.tags
    
    @pytest.mark.asyncio
    async def test_get_performance_trends(self, metrics_repository):
        """Test getting performance trends with time-based aggregation."""
        metric_name = "response_time"
        
        # Create metrics across different hours
        base_time = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        
        for hour in range(3):
            for minute in [0, 30]:  # Two metrics per hour
                timestamp = base_time - timedelta(hours=hour, minutes=minute)
                metric = SystemMetric(
                    metric_name=metric_name,
                    metric_value=Decimal(str(100 + hour * 10 + minute)),
                    instance_id="test-instance",
                    timestamp=timestamp
                )
                await metrics_repository.create(metric)
        
        # Get performance trends
        trends = await metrics_repository.get_performance_trends(
            metric_name, hours=4, interval_minutes=60
        )
        
        assert len(trends) >= 1  # Should have at least one time bucket
        
        for trend in trends:
            assert "timestamp" in trend
            assert "avg_value" in trend
            assert "min_value" in trend
            assert "max_value" in trend
            assert "count" in trend
    
    @pytest.mark.asyncio
    async def test_get_alert_metrics(self, metrics_repository):
        """Test checking metrics against alert thresholds."""
        # Create metrics with different values
        metrics_data = [
            ("cpu_utilization", Decimal('95.0')),  # High value
            ("memory_usage", Decimal('30.0')),     # Low value
            ("disk_usage", Decimal('50.0'))        # Normal value
        ]
        
        for metric_name, value in metrics_data:
            metric = SystemMetric(
                metric_name=metric_name,
                metric_value=value,
                instance_id="test-instance",
                timestamp=datetime.utcnow()
            )
            await metrics_repository.create(metric)
        
        # Define thresholds
        thresholds = {
            "cpu_utilization": {"max": 90.0, "severity": "critical"},
            "memory_usage": {"min": 40.0, "severity": "warning"},
            "disk_usage": {"max": 80.0, "min": 20.0, "severity": "info"}
        }
        
        # Check alerts
        alerts = await metrics_repository.get_alert_metrics(thresholds)
        
        # Should have 2 alerts: CPU too high, memory too low
        assert len(alerts) == 2
        
        alert_metrics = [alert["metric_name"] for alert in alerts]
        assert "cpu_utilization" in alert_metrics
        assert "memory_usage" in alert_metrics
        
        # Check alert details
        cpu_alert = next(a for a in alerts if a["metric_name"] == "cpu_utilization")
        assert cpu_alert["threshold_type"] == "max"
        assert cpu_alert["current_value"] == 95.0
        assert cpu_alert["severity"] == "critical"
    
    @pytest.mark.asyncio
    async def test_get_capacity_metrics(self, metrics_repository):
        """Test getting capacity and utilization metrics for scaling decisions."""
        # Create capacity-related metrics
        capacity_metrics = [
            ("queue_depth_total", Decimal('25.0')),
            ("gpu_utilization_percent", Decimal('85.0')),
            ("memory_usage_percent", Decimal('70.0')),
            ("translation_requests_per_second", Decimal('15.0')),
            ("active_instances_count", Decimal('3.0'))
        ]
        
        for metric_name, value in capacity_metrics:
            metric = SystemMetric(
                metric_name=metric_name,
                metric_value=value,
                instance_id="capacity-instance",
                timestamp=datetime.utcnow()
            )
            await metrics_repository.create(metric)
        
        # Get capacity metrics
        capacity_data = await metrics_repository.get_capacity_metrics(hours=1)
        
        assert "queue_depth" in capacity_data
        assert "gpu_utilization" in capacity_data
        assert "memory_utilization" in capacity_data
        assert "processing_rate" in capacity_data
        assert "active_instances" in capacity_data
        assert "timestamp" in capacity_data
        assert "time_window_hours" in capacity_data
        
        # Check that aggregates contain expected keys
        queue_depth = capacity_data["queue_depth"]
        assert "avg" in queue_depth
        assert "max" in queue_depth
        assert "count" in queue_depth