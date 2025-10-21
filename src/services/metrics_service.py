"""
Metrics collection service with Prometheus integration.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import uuid4

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
import structlog

from src.config.config import config
from src.database.connection import get_db_session
from src.database.repositories import MetricsRepository
from src.database.models import SystemMetric
from src.utils.logging import TranslationLogger

logger = TranslationLogger(__name__, "metrics-service")


class MetricsCollectionService:
    """Prometheus-based metrics collection service."""
    
    def __init__(self):
        # Create custom registry for isolation
        self.registry = CollectorRegistry()
        
        # Translation metrics
        self.translation_requests_total = Counter(
            'translation_requests_total',
            'Total number of translation requests',
            ['source_language', 'target_language', 'priority', 'status'],
            registry=self.registry
        )
        
        self.translation_duration_seconds = Histogram(
            'translation_duration_seconds',
            'Translation processing duration in seconds',
            ['source_language', 'target_language', 'priority'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )
        
        self.translation_words_per_minute = Histogram(
            'translation_words_per_minute',
            'Translation speed in words per minute',
            ['source_language', 'target_language'],
            buckets=[100, 500, 1000, 1500, 2000, 3000, 5000],
            registry=self.registry
        )
        
        self.translation_confidence_score = Histogram(
            'translation_confidence_score',
            'Translation confidence scores',
            ['source_language', 'target_language'],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry
        )
        
        # Queue metrics
        self.queue_depth = Gauge(
            'queue_depth',
            'Current queue depth by priority',
            ['priority'],
            registry=self.registry
        )
        
        self.queue_processing_time = Histogram(
            'queue_processing_time_seconds',
            'Time jobs spend in queue before processing',
            ['priority'],
            buckets=[1, 5, 10, 30, 60, 300, 600, 1800],
            registry=self.registry
        )
        
        # System resource metrics
        self.cpu_utilization = Gauge(
            'cpu_utilization_percent',
            'CPU utilization percentage',
            ['instance_id'],
            registry=self.registry
        )
        
        self.gpu_utilization = Gauge(
            'gpu_utilization_percent',
            'GPU utilization percentage',
            ['instance_id', 'gpu_id'],
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'memory_usage_bytes',
            'Memory usage in bytes',
            ['instance_id', 'type'],
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_hits_total = Counter(
            'cache_hits_total',
            'Total cache hits',
            ['cache_level'],
            registry=self.registry
        )
        
        self.cache_misses_total = Counter(
            'cache_misses_total',
            'Total cache misses',
            ['cache_level'],
            registry=self.registry
        )
        
        self.cache_size = Gauge(
            'cache_size_entries',
            'Current cache size in entries',
            ['cache_level'],
            registry=self.registry
        )
        
        # API metrics
        self.api_requests_total = Counter(
            'api_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.api_request_duration = Histogram(
            'api_request_duration_seconds',
            'API request duration',
            ['method', 'endpoint'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry
        )
        
        # Cost metrics
        self.compute_cost_usd = Counter(
            'compute_cost_usd_total',
            'Total compute cost in USD',
            ['instance_type', 'region'],
            registry=self.registry
        )
        
        self.storage_cost_usd = Counter(
            'storage_cost_usd_total',
            'Total storage cost in USD',
            ['storage_type'],
            registry=self.registry
        )
        
        # Correlation ID for structured logging
        self.correlation_ids: Dict[str, str] = {}
        
        # Background tasks
        self._collection_task = None
        self._running = False
        
        logger.info("Metrics collection service initialized")
    
    async def start(self):
        """Start metrics collection service."""
        if self._running:
            return
        
        self._running = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        
        logger.info("Metrics collection service started")
    
    async def stop(self):
        """Stop metrics collection service."""
        if not self._running:
            return
        
        self._running = False
        
        if self._collection_task and not self._collection_task.done():
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Metrics collection service stopped")
    
    def record_translation_request(
        self,
        source_language: str,
        target_language: str,
        priority: str,
        status: str,
        duration_seconds: float = None,
        word_count: int = None,
        confidence_score: float = None,
        correlation_id: str = None
    ):
        """Record translation request metrics."""
        try:
            # Record request count
            self.translation_requests_total.labels(
                source_language=source_language,
                target_language=target_language,
                priority=priority,
                status=status
            ).inc()
            
            # Record duration if provided
            if duration_seconds is not None:
                self.translation_duration_seconds.labels(
                    source_language=source_language,
                    target_language=target_language,
                    priority=priority
                ).observe(duration_seconds)
            
            # Record words per minute if both duration and word count provided
            if duration_seconds and word_count and duration_seconds > 0:
                wpm = (word_count / duration_seconds) * 60
                self.translation_words_per_minute.labels(
                    source_language=source_language,
                    target_language=target_language
                ).observe(wpm)
            
            # Record confidence score if provided
            if confidence_score is not None:
                self.translation_confidence_score.labels(
                    source_language=source_language,
                    target_language=target_language
                ).observe(confidence_score)
            
            # Structured logging with correlation ID
            log_data = {
                "event": "translation_request",
                "source_language": source_language,
                "target_language": target_language,
                "priority": priority,
                "status": status,
                "duration_seconds": duration_seconds,
                "word_count": word_count,
                "confidence_score": confidence_score
            }
            
            if correlation_id:
                log_data["correlation_id"] = correlation_id
                self.correlation_ids[correlation_id] = datetime.utcnow().isoformat()
            
            logger.info("Translation request recorded", **log_data)
            
        except Exception as e:
            logger.error(f"Failed to record translation metrics: {str(e)}")
    
    def record_queue_metrics(self, priority: str, depth: int, processing_time_seconds: float = None):
        """Record queue metrics."""
        try:
            self.queue_depth.labels(priority=priority).set(depth)
            
            if processing_time_seconds is not None:
                self.queue_processing_time.labels(priority=priority).observe(processing_time_seconds)
            
            logger.debug(
                "Queue metrics recorded",
                priority=priority,
                depth=depth,
                processing_time_seconds=processing_time_seconds
            )
            
        except Exception as e:
            logger.error(f"Failed to record queue metrics: {str(e)}")
    
    def record_system_metrics(
        self,
        instance_id: str,
        cpu_percent: float = None,
        gpu_percent: float = None,
        memory_bytes: int = None,
        gpu_id: str = "0"
    ):
        """Record system resource metrics."""
        try:
            if cpu_percent is not None:
                self.cpu_utilization.labels(instance_id=instance_id).set(cpu_percent)
            
            if gpu_percent is not None:
                self.gpu_utilization.labels(instance_id=instance_id, gpu_id=gpu_id).set(gpu_percent)
            
            if memory_bytes is not None:
                self.memory_usage.labels(instance_id=instance_id, type="total").set(memory_bytes)
            
            logger.debug(
                "System metrics recorded",
                instance_id=instance_id,
                cpu_percent=cpu_percent,
                gpu_percent=gpu_percent,
                memory_bytes=memory_bytes
            )
            
        except Exception as e:
            logger.error(f"Failed to record system metrics: {str(e)}")
    
    def record_cache_metrics(self, cache_level: str, hits: int = None, misses: int = None, size: int = None):
        """Record cache performance metrics."""
        try:
            if hits is not None:
                self.cache_hits_total.labels(cache_level=cache_level).inc(hits)
            
            if misses is not None:
                self.cache_misses_total.labels(cache_level=cache_level).inc(misses)
            
            if size is not None:
                self.cache_size.labels(cache_level=cache_level).set(size)
            
            logger.debug(
                "Cache metrics recorded",
                cache_level=cache_level,
                hits=hits,
                misses=misses,
                size=size
            )
            
        except Exception as e:
            logger.error(f"Failed to record cache metrics: {str(e)}")
    
    def record_api_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration_seconds: float,
        correlation_id: str = None
    ):
        """Record API request metrics."""
        try:
            self.api_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status_code=str(status_code)
            ).inc()
            
            self.api_request_duration.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration_seconds)
            
            # Structured logging
            log_data = {
                "event": "api_request",
                "method": method,
                "endpoint": endpoint,
                "status_code": status_code,
                "duration_seconds": duration_seconds
            }
            
            if correlation_id:
                log_data["correlation_id"] = correlation_id
            
            logger.info("API request recorded", **log_data)
            
        except Exception as e:
            logger.error(f"Failed to record API metrics: {str(e)}")
    
    def record_cost_metrics(self, instance_type: str, region: str, cost_usd: float, storage_type: str = None, storage_cost_usd: float = None):
        """Record cost metrics."""
        try:
            if cost_usd > 0:
                self.compute_cost_usd.labels(
                    instance_type=instance_type,
                    region=region
                ).inc(cost_usd)
            
            if storage_type and storage_cost_usd and storage_cost_usd > 0:
                self.storage_cost_usd.labels(storage_type=storage_type).inc(storage_cost_usd)
            
            logger.debug(
                "Cost metrics recorded",
                instance_type=instance_type,
                region=region,
                cost_usd=cost_usd,
                storage_type=storage_type,
                storage_cost_usd=storage_cost_usd
            )
            
        except Exception as e:
            logger.error(f"Failed to record cost metrics: {str(e)}")
    
    async def get_metrics_export(self) -> str:
        """Get Prometheus metrics export."""
        try:
            return generate_latest(self.registry).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to generate metrics export: {str(e)}")
            return ""
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary for monitoring dashboard."""
        try:
            # This would typically query the metrics from Prometheus
            # For now, return a summary based on current state
            
            summary = {
                "timestamp": datetime.utcnow().isoformat(),
                "translation_metrics": {
                    "total_requests": self._get_counter_value(self.translation_requests_total),
                    "avg_duration_seconds": self._get_histogram_avg(self.translation_duration_seconds),
                    "avg_words_per_minute": self._get_histogram_avg(self.translation_words_per_minute),
                    "avg_confidence_score": self._get_histogram_avg(self.translation_confidence_score)
                },
                "queue_metrics": {
                    "total_depth": self._get_gauge_sum(self.queue_depth),
                    "avg_processing_time": self._get_histogram_avg(self.queue_processing_time)
                },
                "cache_metrics": {
                    "total_hits": self._get_counter_value(self.cache_hits_total),
                    "total_misses": self._get_counter_value(self.cache_misses_total),
                    "hit_rate_percent": self._calculate_cache_hit_rate()
                },
                "api_metrics": {
                    "total_requests": self._get_counter_value(self.api_requests_total),
                    "avg_duration_seconds": self._get_histogram_avg(self.api_request_duration)
                },
                "cost_metrics": {
                    "total_compute_cost_usd": self._get_counter_value(self.compute_cost_usd),
                    "total_storage_cost_usd": self._get_counter_value(self.storage_cost_usd)
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get metrics summary: {str(e)}")
            return {"error": str(e)}
    
    async def store_metrics_to_database(self):
        """Store current metrics to database for historical analysis."""
        try:
            metrics_to_store = []
            current_time = datetime.utcnow()
            
            # Store key metrics to database
            summary = await self.get_metrics_summary()
            
            for category, metrics in summary.items():
                if isinstance(metrics, dict) and category != "timestamp":
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            metric = SystemMetric(
                                timestamp=current_time,
                                metric_name=f"{category}.{metric_name}",
                                metric_value=float(value),
                                tags={"category": category}
                            )
                            metrics_to_store.append(metric)
            
            if metrics_to_store:
                async with get_db_session() as session:
                    metrics_repo = MetricsRepository(session)
                    
                    for metric in metrics_to_store:
                        await metrics_repo.create(metric)
                    
                    await session.commit()
                
                logger.debug(f"Stored {len(metrics_to_store)} metrics to database")
            
        except Exception as e:
            logger.error(f"Failed to store metrics to database: {str(e)}")
    
    def generate_correlation_id(self) -> str:
        """Generate correlation ID for request tracking."""
        correlation_id = str(uuid4())
        self.correlation_ids[correlation_id] = datetime.utcnow().isoformat()
        return correlation_id
    
    def get_correlation_context(self, correlation_id: str) -> Optional[str]:
        """Get correlation context timestamp."""
        return self.correlation_ids.get(correlation_id)
    
    def cleanup_correlation_ids(self, max_age_hours: int = 24):
        """Clean up old correlation IDs."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
            
            expired_ids = []
            for correlation_id, timestamp_str in self.correlation_ids.items():
                timestamp = datetime.fromisoformat(timestamp_str)
                if timestamp < cutoff_time:
                    expired_ids.append(correlation_id)
            
            for correlation_id in expired_ids:
                del self.correlation_ids[correlation_id]
            
            if expired_ids:
                logger.debug(f"Cleaned up {len(expired_ids)} expired correlation IDs")
            
        except Exception as e:
            logger.error(f"Failed to cleanup correlation IDs: {str(e)}")
    
    async def _collection_loop(self):
        """Background metrics collection loop."""
        while self._running:
            try:
                # Store metrics to database every 5 minutes
                await self.store_metrics_to_database()
                
                # Cleanup correlation IDs every hour
                self.cleanup_correlation_ids()
                
                await asyncio.sleep(300)  # 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {str(e)}")
                await asyncio.sleep(300)
    
    def _get_counter_value(self, counter) -> float:
        """Get total value from counter metric."""
        try:
            total = 0.0
            for sample in counter.collect()[0].samples:
                total += sample.value
            return total
        except:
            return 0.0
    
    def _get_histogram_avg(self, histogram) -> float:
        """Get average value from histogram metric."""
        try:
            total_sum = 0.0
            total_count = 0.0
            
            for sample in histogram.collect()[0].samples:
                if sample.name.endswith('_sum'):
                    total_sum += sample.value
                elif sample.name.endswith('_count'):
                    total_count += sample.value
            
            return total_sum / total_count if total_count > 0 else 0.0
        except:
            return 0.0
    
    def _get_gauge_sum(self, gauge) -> float:
        """Get sum of all gauge values."""
        try:
            total = 0.0
            for sample in gauge.collect()[0].samples:
                total += sample.value
            return total
        except:
            return 0.0
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate overall cache hit rate."""
        try:
            total_hits = self._get_counter_value(self.cache_hits_total)
            total_misses = self._get_counter_value(self.cache_misses_total)
            total_requests = total_hits + total_misses
            
            return (total_hits / total_requests * 100) if total_requests > 0 else 0.0
        except:
            return 0.0


def create_metrics_service() -> MetricsCollectionService:
    """Factory function to create metrics service."""
    return MetricsCollectionService()