"""
Custom exceptions for the machine translation system.
"""

from typing import Optional, Dict, Any


class TranslationSystemException(Exception):
    """Base exception for translation system errors."""
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "SYSTEM_ERROR"
        self.details = details or {}


class ValidationError(TranslationSystemException):
    """Exception for request validation errors."""
    
    def __init__(self, message: str, field: str = None, details: Dict[str, Any] = None):
        super().__init__(message, "VALIDATION_ERROR", details)
        self.field = field


class AuthenticationError(TranslationSystemException):
    """Exception for authentication failures."""
    
    def __init__(self, message: str = "Authentication failed", details: Dict[str, Any] = None):
        super().__init__(message, "AUTHENTICATION_ERROR", details)


class AuthorizationError(TranslationSystemException):
    """Exception for authorization failures."""
    
    def __init__(self, message: str = "Access denied", details: Dict[str, Any] = None):
        super().__init__(message, "AUTHORIZATION_ERROR", details)


class RateLimitError(TranslationSystemException):
    """Exception for rate limit violations."""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = None, details: Dict[str, Any] = None):
        super().__init__(message, "RATE_LIMIT_ERROR", details)
        self.retry_after = retry_after


class UnsupportedLanguageError(TranslationSystemException):
    """Exception for unsupported language pairs."""
    
    def __init__(self, source_lang: str, target_lang: str, supported_languages: list = None):
        message = f"Translation from '{source_lang}' to '{target_lang}' is not supported"
        details = {"source_language": source_lang, "target_language": target_lang}
        if supported_languages:
            details["supported_languages"] = supported_languages
        super().__init__(message, "UNSUPPORTED_LANGUAGE", details)


class TranslationEngineError(TranslationSystemException):
    """Exception for translation engine failures."""
    
    def __init__(self, message: str, engine_id: str = None, details: Dict[str, Any] = None):
        super().__init__(message, "TRANSLATION_ENGINE_ERROR", details)
        self.engine_id = engine_id


class ModelLoadError(TranslationEngineError):
    """Exception for model loading failures."""
    
    def __init__(self, model_id: str, message: str = None, details: Dict[str, Any] = None):
        message = message or f"Failed to load model: {model_id}"
        super().__init__(message, "MODEL_LOAD_ERROR", details)
        self.model_id = model_id


class InsufficientResourcesError(TranslationSystemException):
    """Exception for insufficient compute resources."""
    
    def __init__(self, resource_type: str, required: float, available: float, details: Dict[str, Any] = None):
        message = f"Insufficient {resource_type}: required {required}, available {available}"
        details = details or {}
        details.update({
            "resource_type": resource_type,
            "required": required,
            "available": available
        })
        super().__init__(message, "INSUFFICIENT_RESOURCES", details)


class QueueError(TranslationSystemException):
    """Exception for queue management errors."""
    
    def __init__(self, message: str, queue_name: str = None, details: Dict[str, Any] = None):
        super().__init__(message, "QUEUE_ERROR", details)
        self.queue_name = queue_name


class CacheError(TranslationSystemException):
    """Exception for cache operations."""
    
    def __init__(self, message: str, cache_key: str = None, details: Dict[str, Any] = None):
        super().__init__(message, "CACHE_ERROR", details)
        self.cache_key = cache_key


class DatabaseError(TranslationSystemException):
    """Exception for database operations."""
    
    def __init__(self, message: str, operation: str = None, details: Dict[str, Any] = None):
        super().__init__(message, "DATABASE_ERROR", details)
        self.operation = operation


class JobNotFoundError(TranslationSystemException):
    """Exception for job lookup failures."""
    
    def __init__(self, job_id: str, details: Dict[str, Any] = None):
        message = f"Translation job not found: {job_id}"
        super().__init__(message, "JOB_NOT_FOUND", details)
        self.job_id = job_id


class JobTimeoutError(TranslationSystemException):
    """Exception for job processing timeouts."""
    
    def __init__(self, job_id: str, timeout_seconds: int, details: Dict[str, Any] = None):
        message = f"Translation job timed out after {timeout_seconds} seconds: {job_id}"
        details = details or {}
        details.update({
            "job_id": job_id,
            "timeout_seconds": timeout_seconds
        })
        super().__init__(message, "JOB_TIMEOUT", details)


class ResourceScalingError(TranslationSystemException):
    """Exception for resource scaling failures."""
    
    def __init__(self, action: str, message: str = None, details: Dict[str, Any] = None):
        message = message or f"Resource scaling failed: {action}"
        super().__init__(message, "RESOURCE_SCALING_ERROR", details)
        self.action = action


class MonitoringError(TranslationSystemException):
    """Exception for monitoring system failures."""
    
    def __init__(self, message: str, metric_name: str = None, details: Dict[str, Any] = None):
        super().__init__(message, "MONITORING_ERROR", details)
        self.metric_name = metric_name


class ConfigurationError(TranslationSystemException):
    """Exception for configuration errors."""
    
    def __init__(self, message: str, config_key: str = None, details: Dict[str, Any] = None):
        super().__init__(message, "CONFIGURATION_ERROR", details)
        self.config_key = config_key


class CircuitBreakerError(TranslationSystemException):
    """Exception for circuit breaker activation."""
    
    def __init__(self, service_name: str, message: str = None, details: Dict[str, Any] = None):
        message = message or f"Circuit breaker open for service: {service_name}"
        super().__init__(message, "CIRCUIT_BREAKER_OPEN", details)
        self.service_name = service_name


def create_error_response(exception: TranslationSystemException, include_details: bool = True) -> Dict[str, Any]:
    """Create standardized error response from exception."""
    response = {
        "error": {
            "code": exception.error_code,
            "message": exception.message
        }
    }
    
    if include_details and exception.details:
        response["error"]["details"] = exception.details
    
    # Add retry_after for rate limit errors
    if isinstance(exception, RateLimitError) and exception.retry_after:
        response["retry_after"] = exception.retry_after
    
    return response


# Aliases for backward compatibility
TranslationError = TranslationEngineError
ResourceError = InsufficientResourcesError