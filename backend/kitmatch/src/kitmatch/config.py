"""
Application Configuration
This module centralizes all configuration for the kitmatch application.
It uses Pydantic's BaseSettings to load settings from environment variables
and a .env file, providing validation and type hints.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
import os


PROJECT_DIRECTORY = Path(__file__).parent.parent

# RabbitMQ Configuration
RABBIT_HEARTBEAT_MODELS = int(os.getenv("RABBIT_HEARTBEAT_MODELS", "600"))
RABBIT_BLOCKED_CONN_TIMEOUT_MODELS = int(os.getenv("RABBIT_BLOCKED_CONN_TIMEOUT_MODELS", "300"))

RABBIT_HEARTBEAT_REQUESTS = int(os.getenv("RABBIT_HEARTBEAT_REQUESTS", "3000")) # max 5 minutes per banner, max 10 banners => 50 mins
RABBIT_BLOCKED_CONN_TIMEOUT_REQUESTS = int(os.getenv("RABBIT_BLOCKED_CONN_TIMEOUT_REQUESTS", "3000"))

# Models Configuration
MODELS_URI = os.getenv("MODELS_URI", "10.5.2.86")
MODELS_PORT = os.getenv("MODELS_PORT", "1339")
MODELS_ADDRESS = f"{MODELS_URI}:{MODELS_PORT}"


class Settings(BaseSettings):
    """
    Настройки приложения, загружаемые из окружения и .env файла.

    Атрибуты:
        app_env (str): Среда выполнения приложения (например, 'dev', 'prod').
        api_host (str): Хост для запуска API сервера.
        api_port (int): Порт для запуска API сервера.
        log_level (str): Уровень логирования.
        log_json (bool): Форматировать ли логи в JSON.
    """

    app_env: str = "prod"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    log_json: bool = False

    # Triton and static
    triton_url: str = "http://localhost:1339"
    triton_model_name: str = "yolo"
    triton_model_version: str = "1"
    static_dir: str = "static"
    static_url: str = "/static"

    # MongoDB Configuration
    mongodb_uri: str = "mongo_host"
    mongodb_port: int = 27017
    mongodb_username: str = "root"
    mongodb_password: str = "example"
    mongodb_db_name: str = "kitmatch"
    mongodb_collection_name: str = "detections"

    # RabbitMQ Configuration
    rabbitmq_host: str = "rabbit_host"

    model_config = SettingsConfigDict(
        env_prefix="BACKEND__",
        env_nested_delimiter="__",
        env_file=PROJECT_DIRECTORY / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


# Создаем единственный экземпляр настроек для использования во всем приложении
settings = Settings()
