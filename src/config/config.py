"""
Configuration management for the machine translation backend system.
Supports different environments (development, staging, production).
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Priority(Enum):
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class JobStatus(Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class DatabaseConfig:
    host: str
    port: int
    database: str
    username: str
    password: str
    pool_size: int = 10
    max_overflow: int = 20
    
    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class RedisConfig:
    host: str
    port: int
    password: Optional[str] = None
    db: int = 0
    max_connections: int = 100


@dataclass
class ComputeConfig:
    gpu_instance_type: str = "g4dn.xlarge"
    cpu_instance_type: str = "r5.8xlarge"
    min_instances: int = 1
    max_instances: int = 10
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.2
    idle_timeout_minutes: int = 10


@dataclass
class PerformanceConfig:
    target_words_per_minute: int = 1500
    max_document_words: int = 15000
    max_processing_time_minutes: int = 12
    cache_ttl_hours: int = 24
    model_memory_threshold: float = 0.85


@dataclass
class SecurityConfig:
    jwt_secret_key: str
    encryption_key: str
    jwt_expiration_hours: int = 24
    rate_limit_per_minute: int = 100
    api_key_length: int = 32


@dataclass
class MonitoringConfig:
    prometheus_port: int = 9090
    log_level: str = "INFO"
    alert_webhook_url: Optional[str] = None
    cost_alert_threshold_usd: float = 1000.0


@dataclass
class TranslationConfig:
    use_gpu: bool = True
    require_gpu: bool = False
    max_models_in_memory: int = 3
    max_memory_usage_percent: float = 85.0
    batch_size: int = 8
    max_sequence_length: int = 512
    model_cache_dir: str = "./models"
    supported_languages: List[str] = None
    model_configs: Dict[str, Dict] = None
    
    def __post_init__(self):
        if self.supported_languages is None:
            self.supported_languages = ["en", "es", "fr", "de", "zh", "ja", "ko", "pt", "it", "ru"]
        
        if self.model_configs is None:
            self.model_configs = {
                "multilingual-base": {
                    "model_name": "facebook/mbart-large-50-many-to-many-mmt",
                    "version": "1.0.0",
                    "multilingual": True,
                    "language_pairs": ["en-es", "en-fr", "en-de", "es-en", "fr-en", "de-en"],
                    "requires_language_prefix": False,
                    "requires_target_prefix": True
                },
                "en-es-optimized": {
                    "model_name": "Helsinki-NLP/opus-mt-en-es",
                    "version": "1.0.0",
                    "multilingual": False,
                    "language_pairs": ["en-es"],
                    "requires_language_prefix": False,
                    "requires_target_prefix": False
                },
                "es-en-optimized": {
                    "model_name": "Helsinki-NLP/opus-mt-es-en",
                    "version": "1.0.0",
                    "multilingual": False,
                    "language_pairs": ["es-en"],
                    "requires_language_prefix": False,
                    "requires_target_prefix": False
                }
            }


@dataclass
class Config:
    environment: Environment
    database: DatabaseConfig
    redis: RedisConfig
    compute: ComputeConfig
    performance: PerformanceConfig
    security: SecurityConfig
    monitoring: MonitoringConfig
    translation: TranslationConfig
    
    supported_languages: List[str] = None
    
    def __post_init__(self):
        if self.supported_languages is None:
            self.supported_languages = ["en", "es", "fr", "de", "zh", "ja", "ko", "pt", "it", "ru"]


def load_config() -> Config:
    """Load configuration based on environment variables."""
    env = Environment(os.getenv("ENVIRONMENT", "development"))
    
    database_config = DatabaseConfig(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_NAME", "translation_db"),
        username=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "password"),
        pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
        max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20"))
    )
    
    redis_config = RedisConfig(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        password=os.getenv("REDIS_PASSWORD"),
        db=int(os.getenv("REDIS_DB", "0")),
        max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "100"))
    )
    
    compute_config = ComputeConfig(
        gpu_instance_type=os.getenv("GPU_INSTANCE_TYPE", "g4dn.xlarge"),
        cpu_instance_type=os.getenv("CPU_INSTANCE_TYPE", "r5.8xlarge"),
        min_instances=int(os.getenv("MIN_INSTANCES", "1")),
        max_instances=int(os.getenv("MAX_INSTANCES", "10")),
        scale_up_threshold=float(os.getenv("SCALE_UP_THRESHOLD", "0.8")),
        scale_down_threshold=float(os.getenv("SCALE_DOWN_THRESHOLD", "0.2")),
        idle_timeout_minutes=int(os.getenv("IDLE_TIMEOUT_MINUTES", "10"))
    )
    
    performance_config = PerformanceConfig(
        target_words_per_minute=int(os.getenv("TARGET_WPM", "1500")),
        max_document_words=int(os.getenv("MAX_DOCUMENT_WORDS", "15000")),
        max_processing_time_minutes=int(os.getenv("MAX_PROCESSING_TIME", "12")),
        cache_ttl_hours=int(os.getenv("CACHE_TTL_HOURS", "24")),
        model_memory_threshold=float(os.getenv("MODEL_MEMORY_THRESHOLD", "0.85"))
    )
    
    security_config = SecurityConfig(
        jwt_secret_key=os.getenv("JWT_SECRET_KEY", "dev-secret-key"),
        jwt_expiration_hours=int(os.getenv("JWT_EXPIRATION_HOURS", "24")),
        rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "100")),
        encryption_key=os.getenv("ENCRYPTION_KEY", "dev-encryption-key"),
        api_key_length=int(os.getenv("API_KEY_LENGTH", "32"))
    )
    
    monitoring_config = MonitoringConfig(
        prometheus_port=int(os.getenv("PROMETHEUS_PORT", "9090")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        alert_webhook_url=os.getenv("ALERT_WEBHOOK_URL"),
        cost_alert_threshold_usd=float(os.getenv("COST_ALERT_THRESHOLD", "1000.0"))
    )
    
    translation_config = TranslationConfig(
        use_gpu=os.getenv("USE_GPU", "true").lower() == "true",
        require_gpu=os.getenv("REQUIRE_GPU", "false").lower() == "true",
        max_models_in_memory=int(os.getenv("MAX_MODELS_IN_MEMORY", "3")),
        max_memory_usage_percent=float(os.getenv("MAX_MEMORY_USAGE_PERCENT", "85.0")),
        batch_size=int(os.getenv("TRANSLATION_BATCH_SIZE", "8")),
        max_sequence_length=int(os.getenv("MAX_SEQUENCE_LENGTH", "512")),
        model_cache_dir=os.getenv("MODEL_CACHE_DIR", "./models")
    )
    
    return Config(
        environment=env,
        database=database_config,
        redis=redis_config,
        compute=compute_config,
        performance=performance_config,
        security=security_config,
        monitoring=monitoring_config,
        translation=translation_config
    )


# Global configuration instance
config = load_config()