"""
Structured logging utilities for the machine translation system.
"""

import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

from src.config.config import config


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "service": getattr(record, 'service', 'translation-system'),
            "instance_id": getattr(record, 'instance_id', None),
            "job_id": getattr(record, 'job_id', None),
            "event": getattr(record, 'event', record.funcName),
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add metrics if present
        if hasattr(record, 'metrics'):
            log_entry["metrics"] = record.metrics
            
        # Add metadata if present
        if hasattr(record, 'metadata'):
            log_entry["metadata"] = record.metadata
            
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry, default=str)


class TranslationLogger:
    """Enhanced logger for translation system with structured logging."""
    
    def __init__(self, name: str, service: str = "translation-system"):
        self.logger = logging.getLogger(name)
        self.service = service
        self._setup_logger()
    
    def _setup_logger(self):
        """Configure logger with structured formatting."""
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(StructuredFormatter())
            self.logger.addHandler(handler)
            self.logger.setLevel(getattr(logging, config.monitoring.log_level))
    
    def info(self, message: str, job_id: Optional[UUID] = None, 
             instance_id: Optional[str] = None, event: Optional[str] = None,
             metrics: Optional[Dict[str, Any]] = None, 
             metadata: Optional[Dict[str, Any]] = None):
        """Log info level message with structured data."""
        extra = {
            'service': self.service,
            'job_id': str(job_id) if job_id else None,
            'instance_id': instance_id,
            'event': event,
            'metrics': metrics,
            'metadata': metadata
        }
        self.logger.info(message, extra=extra)
    
    def warning(self, message: str, job_id: Optional[UUID] = None,
                instance_id: Optional[str] = None, event: Optional[str] = None,
                metrics: Optional[Dict[str, Any]] = None,
                metadata: Optional[Dict[str, Any]] = None):
        """Log warning level message with structured data."""
        extra = {
            'service': self.service,
            'job_id': str(job_id) if job_id else None,
            'instance_id': instance_id,
            'event': event,
            'metrics': metrics,
            'metadata': metadata
        }
        self.logger.warning(message, extra=extra)
    
    def error(self, message: str, job_id: Optional[UUID] = None,
              instance_id: Optional[str] = None, event: Optional[str] = None,
              metrics: Optional[Dict[str, Any]] = None,
              metadata: Optional[Dict[str, Any]] = None, exc_info: bool = False):
        """Log error level message with structured data."""
        extra = {
            'service': self.service,
            'job_id': str(job_id) if job_id else None,
            'instance_id': instance_id,
            'event': event,
            'metrics': metrics,
            'metadata': metadata
        }
        self.logger.error(message, extra=extra, exc_info=exc_info)
    
    def debug(self, message: str, job_id: Optional[UUID] = None,
              instance_id: Optional[str] = None, event: Optional[str] = None,
              metrics: Optional[Dict[str, Any]] = None,
              metadata: Optional[Dict[str, Any]] = None):
        """Log debug level message with structured data."""
        extra = {
            'service': self.service,
            'job_id': str(job_id) if job_id else None,
            'instance_id': instance_id,
            'event': event,
            'metrics': metrics,
            'metadata': metadata
        }
        self.logger.debug(message, extra=extra)
    
    def translation_started(self, job_id: UUID, instance_id: str, 
                          source_lang: str, target_lang: str, word_count: int):
        """Log translation job start."""
        self.info(
            f"Translation job started",
            job_id=job_id,
            instance_id=instance_id,
            event="translation_started",
            metadata={
                "source_language": source_lang,
                "target_language": target_lang,
                "word_count": word_count
            }
        )
    
    def translation_completed(self, job_id: UUID, instance_id: str,
                            processing_time_ms: int, word_count: int,
                            words_per_minute: float, gpu_utilization: float,
                            memory_usage_mb: float, model_version: str,
                            source_lang: str, target_lang: str, 
                            priority: str, cache_hit: bool):
        """Log translation job completion."""
        self.info(
            f"Translation job completed",
            job_id=job_id,
            instance_id=instance_id,
            event="translation_completed",
            metrics={
                "processing_time_ms": processing_time_ms,
                "word_count": word_count,
                "words_per_minute": words_per_minute,
                "gpu_utilization": gpu_utilization,
                "memory_usage_mb": memory_usage_mb,
                "model_version": model_version
            },
            metadata={
                "source_language": source_lang,
                "target_language": target_lang,
                "priority": priority,
                "cache_hit": cache_hit
            }
        )
    
    def translation_failed(self, job_id: UUID, instance_id: str,
                          error_message: str, processing_time_ms: int,
                          source_lang: str, target_lang: str):
        """Log translation job failure."""
        self.error(
            f"Translation job failed: {error_message}",
            job_id=job_id,
            instance_id=instance_id,
            event="translation_failed",
            metrics={
                "processing_time_ms": processing_time_ms
            },
            metadata={
                "source_language": source_lang,
                "target_language": target_lang,
                "error_message": error_message
            }
        )
    
    def queue_metrics(self, instance_id: str, queue_depths: Dict[str, int],
                     processing_rate: float, avg_wait_time: float):
        """Log queue performance metrics."""
        self.info(
            "Queue performance metrics",
            instance_id=instance_id,
            event="queue_metrics",
            metrics={
                "critical_queue_depth": queue_depths.get("critical", 0),
                "high_queue_depth": queue_depths.get("high", 0),
                "normal_queue_depth": queue_depths.get("normal", 0),
                "total_queue_depth": sum(queue_depths.values()),
                "processing_rate_jobs_per_minute": processing_rate,
                "avg_wait_time_seconds": avg_wait_time
            }
        )
    
    def resource_scaling(self, event_type: str, instance_count: int,
                        trigger_metric: str, trigger_value: float,
                        threshold: float):
        """Log resource scaling events."""
        self.info(
            f"Resource scaling: {event_type}",
            event="resource_scaling",
            metrics={
                "instance_count": instance_count,
                "trigger_value": trigger_value,
                "threshold": threshold
            },
            metadata={
                "scaling_action": event_type,
                "trigger_metric": trigger_metric
            }
        )
    
    def cache_performance(self, instance_id: str, hit_ratio: float,
                         total_requests: int, cache_size_mb: float,
                         evictions: int):
        """Log cache performance metrics."""
        self.info(
            "Cache performance metrics",
            instance_id=instance_id,
            event="cache_performance",
            metrics={
                "hit_ratio": hit_ratio,
                "total_requests": total_requests,
                "cache_size_mb": cache_size_mb,
                "evictions": evictions
            }
        )


# Global logger instances
api_logger = TranslationLogger("api", "api-gateway")
engine_logger = TranslationLogger("engine", "translation-engine")
queue_logger = TranslationLogger("queue", "queue-manager")
cache_logger = TranslationLogger("cache", "cache-manager")
resource_logger = TranslationLogger("resource", "resource-manager")
monitoring_logger = TranslationLogger("monitoring", "monitoring-service")