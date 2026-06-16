"""
ForgeMind Core Configuration.

Centralized settings management using Pydantic V2 Settings.
All configuration is loaded from environment variables or .env file.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    app_env: str = "development"
    app_debug: bool = True
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # PostgreSQL
    postgres_user: str = "forgemind"
    postgres_password: str = "forgemind_secret"
    postgres_db: str = "forgemind"
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    database_url: str = "postgresql://forgemind:forgemind_secret@postgres:5432/forgemind"

    # MinIO
    minio_root_user: str = "forgemind"
    minio_root_password: str = "forgemind_secret"
    minio_endpoint: str = "minio:9000"
    minio_bucket: str = "forgemind-datasets"
    minio_use_ssl: bool = False

    # Redis
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_url: str = "redis://redis:6379/0"

    # Celery
    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/1"

    # MLflow
    mlflow_tracking_uri: str = "http://mlflow:5000"

    # Ollama (Local LLM Backend)
    ollama_base_url: str = "http://ollama:11434"
    ollama_default_model: str = "qwen2.5:0.5b"


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
