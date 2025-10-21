"""
Repository for system metrics operations.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from uuid import UUID

from sqlalchemy import and_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import SystemMetric
from src.database.repositories.base import BaseRepository, RepositoryMixin
from src.models.interfaces import MetricsRepository as IMetricsRepository
from src.utils.exceptions import DatabaseError
from src.utils.logging import TranslationLogger

logger = TranslationLogger(__name__, "metrics-repository")


class MetricsRepository(BaseRepository[SystemMetric], RepositoryMixin, IMetricsRepository):
    """Repository for system metrics operations."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, SystemMetric)
    
    async def get_metrics(self, metric_name: str, start_time: datetime, 
                         end_time: datetime, instance_id: Optional[str] = None) -> List[SystemMetric]:
        """Get metrics within time range."""
        try:
            filters = {
                'metric_name': metric_name,
                'timestamp': {'gte': start_time, 'lte': end_time}
            }
            
            if instance_id:
                filters['instance_id'] = instance_id
            
            stmt = select(self.model_class)
            stmt = self._apply_filters(stmt, filters)
            stmt = stmt.order_by(self.model_class.timestamp)
            
            result = await self.session.execute(stmt)
            return list(result.scalars().all())
            
        except Exception as e:
            logger.error(f"Error getting metrics for {metric_name}: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to get metrics: {str(e)}")
    
    async def get_latest_metric(self, metric_name: str, 
                              instance_id: Optional[str] = None) -> Optional[SystemMetric]:
        """Get latest metric value."""
        try:
            filters = {'metric_name': metric_name}
            if instance_id:
                filters['instance_id'] = instance_id
            
            stmt = select(self.model_class)
            stmt = self._apply_filters(stmt, filters)
            stmt = stmt.order_by(desc(self.model_class.timestamp)).limit(1)
            
            result = await self.session.execute(stmt)
            return result.scalars().first()
            
        except Exception as e:
            logger.error(f"Error getting latest metric {metric_name}: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to get latest metric: {str(e)}")
    
    async def get_metric_aggregates(self, metric_name: str, start_time: datetime, 
                                  end_time: datetime, instance_id: Optional[str] = None) -> Dict[str, float]:
        """Get aggregated metrics (min, max, avg, sum) for a time range."""
        try:
            base_query = select(self.model_class).where(
                and_(
                    self.model_class.metric_name == metric_name,
                    self.model_class.timestamp >= start_time,
                    self.model_class.timestamp <= end_time
                )
            )
            
            if instance_id:
                base_query = base_query.where(self.model_class.instance_id == instance_id)
            
            # Get aggregates
            aggregates_query = select(
                func.min(self.model_class.metric_value).label('min_value'),
                func.max(self.model_class.metric_value).label('max_value'),
                func.avg(self.model_class.metric_value).label('avg_value'),
                func.sum(self.model_class.metric_value).label('sum_value'),
                func.count().label('count')
            ).select_from(base_query.subquery())
            
            result = await self.session.execute(aggregates_query)
            row = result.first()
            
            if not row or row.count == 0:
                return {
                    'min': 0.0,
                    'max': 0.0,
                    'avg': 0.0,
                    'sum': 0.0,
                    'count': 0
                }
            
            return {
                'min': float(row.min_value or 0),
                'max': float(row.max_value or 0),
                'avg': float(row.avg_value or 0),
                'sum': float(row.sum_value or 0),
                'count': row.count
            }
            
        except Exception as e:
            logger.error(f"Error getting metric aggregates for {metric_name}: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to get metric aggregates: {str(e)}")
    
    async def get_metrics_by_instance(self, instance_id: str, hours: int = 24) -> List[SystemMetric]:
        """Get all metrics for a specific instance."""
        try:
            start_time = datetime.utcnow() - timedelta(hours=hours)
            
            filters = {
                'instance_id': instance_id,
                'timestamp': {'gte': start_time}
            }
            
            stmt = select(self.model_class)
            stmt = self._apply_filters(stmt, filters)
            stmt = stmt.order_by(self.model_class.timestamp)
            
            result = await self.session.execute(stmt)
            return list(result.scalars().all())
            
        except Exception as e:
            logger.error(f"Error getting metrics for instance {instance_id}: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to get metrics for instance: {str(e)}")
    
    async def get_metric_names(self) -> List[str]:
        """Get all unique metric names."""
        try:
            stmt = select(self.model_class.metric_name).distinct()
            result = await self.session.execute(stmt)
            return [row.metric_name for row in result]
        except Exception as e:
            logger.error(f"Error getting metric names: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to get metric names: {str(e)}")
    
    async def get_instance_ids(self) -> List[str]:
        """Get all unique instance IDs."""
        try:
            stmt = select(self.model_class.instance_id).distinct().where(
                self.model_class.instance_id.isnot(None)
            )
            result = await self.session.execute(stmt)
            return [row.instance_id for row in result]
        except Exception as e:
            logger.error(f"Error getting instance IDs: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to get instance IDs: {str(e)}")
    
    async def cleanup_old_metrics(self, days: int = 30) -> int:
        """Clean up old metrics data."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            
            filters = {
                'timestamp': {'lt': cutoff_time}
            }
            
            deleted_count = await self.bulk_delete(filters)
            
            logger.info(f"Cleaned up {deleted_count} old metrics older than {days} days")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old metrics: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to cleanup old metrics: {str(e)}")
    
    async def get_system_health_metrics(self, minutes: int = 5) -> Dict[str, any]:
        """Get recent system health metrics."""
        try:
            start_time = datetime.utcnow() - timedelta(minutes=minutes)
            
            # Key metrics to check
            health_metrics = [
                'gpu_utilization_percent',
                'memory_usage_percent',
                'cpu_utilization_percent',
                'translation_latency_seconds',
                'queue_depth_total',
                'cache_hit_ratio',
                'active_translation_jobs'
            ]
            
            health_data = {}
            
            for metric_name in health_metrics:
                aggregates = await self.get_metric_aggregates(
                    metric_name, start_time, datetime.utcnow()
                )
                health_data[metric_name] = aggregates
            
            # Get instance-specific metrics
            instance_ids = await self.get_instance_ids()
            instance_health = {}
            
            for instance_id in instance_ids:
                instance_metrics = {}
                for metric_name in ['gpu_utilization_percent', 'memory_usage_percent', 'cpu_utilization_percent']:
                    latest = await self.get_latest_metric(metric_name, instance_id)
                    instance_metrics[metric_name] = {
                        'value': float(latest.metric_value) if latest else 0.0,
                        'timestamp': latest.timestamp.isoformat() if latest else None
                    }
                instance_health[instance_id] = instance_metrics
            
            return {
                'system_metrics': health_data,
                'instance_metrics': instance_health,
                'timestamp': datetime.utcnow().isoformat(),
                'time_window_minutes': minutes
            }
            
        except Exception as e:
            logger.error(f"Error getting system health metrics: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to get system health metrics: {str(e)}")
    
    async def record_bulk_metrics(self, metrics: List[Dict[str, any]]) -> bool:
        """Record multiple metrics in bulk."""
        try:
            metric_objects = []
            
            for metric_data in metrics:
                metric = SystemMetric(
                    timestamp=metric_data.get('timestamp', datetime.utcnow()),
                    metric_name=metric_data['metric_name'],
                    metric_value=metric_data['metric_value'],
                    instance_id=metric_data.get('instance_id'),
                    tags=metric_data.get('tags')
                )
                metric_objects.append(metric)
            
            await self.bulk_create(metric_objects)
            return True
            
        except Exception as e:
            logger.error(f"Error recording bulk metrics: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to record bulk metrics: {str(e)}")
    
    async def get_performance_trends(self, metric_name: str, hours: int = 24, 
                                   interval_minutes: int = 60) -> List[Dict[str, any]]:
        """Get performance trends with time-based aggregation."""
        try:
            start_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Create time buckets
            interval_seconds = interval_minutes * 60
            
            # PostgreSQL-specific query for time bucketing
            stmt = select(
                func.date_trunc('hour', self.model_class.timestamp).label('time_bucket'),
                func.avg(self.model_class.metric_value).label('avg_value'),
                func.min(self.model_class.metric_value).label('min_value'),
                func.max(self.model_class.metric_value).label('max_value'),
                func.count().label('count')
            ).where(
                and_(
                    self.model_class.metric_name == metric_name,
                    self.model_class.timestamp >= start_time
                )
            ).group_by(
                func.date_trunc('hour', self.model_class.timestamp)
            ).order_by('time_bucket')
            
            result = await self.session.execute(stmt)
            
            trends = []
            for row in result:
                trends.append({
                    'timestamp': row.time_bucket.isoformat(),
                    'avg_value': float(row.avg_value),
                    'min_value': float(row.min_value),
                    'max_value': float(row.max_value),
                    'count': row.count
                })
            
            return trends
            
        except Exception as e:
            logger.error(f"Error getting performance trends for {metric_name}: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to get performance trends: {str(e)}")
    
    async def get_alert_metrics(self, thresholds: Dict[str, Dict[str, float]]) -> List[Dict[str, any]]:
        """Check metrics against alert thresholds."""
        try:
            alerts = []
            
            for metric_name, threshold_config in thresholds.items():
                latest = await self.get_latest_metric(metric_name)
                
                if not latest:
                    continue
                
                value = float(latest.metric_value)
                
                # Check various threshold types
                if 'max' in threshold_config and value > threshold_config['max']:
                    alerts.append({
                        'metric_name': metric_name,
                        'current_value': value,
                        'threshold_value': threshold_config['max'],
                        'threshold_type': 'max',
                        'severity': threshold_config.get('severity', 'warning'),
                        'instance_id': latest.instance_id,
                        'timestamp': latest.timestamp.isoformat()
                    })
                
                if 'min' in threshold_config and value < threshold_config['min']:
                    alerts.append({
                        'metric_name': metric_name,
                        'current_value': value,
                        'threshold_value': threshold_config['min'],
                        'threshold_type': 'min',
                        'severity': threshold_config.get('severity', 'warning'),
                        'instance_id': latest.instance_id,
                        'timestamp': latest.timestamp.isoformat()
                    })
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking alert metrics: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to check alert metrics: {str(e)}")
    
    async def get_capacity_metrics(self, hours: int = 1) -> Dict[str, any]:
        """Get capacity and utilization metrics for scaling decisions."""
        try:
            start_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Get queue depth metrics
            queue_metrics = await self.get_metric_aggregates(
                'queue_depth_total', start_time, datetime.utcnow()
            )
            
            # Get GPU utilization across all instances
            gpu_metrics = await self.get_metric_aggregates(
                'gpu_utilization_percent', start_time, datetime.utcnow()
            )
            
            # Get memory utilization
            memory_metrics = await self.get_metric_aggregates(
                'memory_usage_percent', start_time, datetime.utcnow()
            )
            
            # Get processing rate
            processing_rate_metrics = await self.get_metric_aggregates(
                'translation_requests_per_second', start_time, datetime.utcnow()
            )
            
            # Get active instances count
            active_instances = await self.get_latest_metric('active_instances_count')
            
            return {
                'queue_depth': queue_metrics,
                'gpu_utilization': gpu_metrics,
                'memory_utilization': memory_metrics,
                'processing_rate': processing_rate_metrics,
                'active_instances': float(active_instances.metric_value) if active_instances else 0,
                'timestamp': datetime.utcnow().isoformat(),
                'time_window_hours': hours
            }
            
        except Exception as e:
            logger.error(f"Error getting capacity metrics: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to get capacity metrics: {str(e)}")