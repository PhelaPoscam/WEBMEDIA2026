"""Application configuration management.

Uses pydantic-settings for environment variable management.
Create a .env file in project root for local development.
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

    # Core metadata
    app_name: str = "ASA"
    app_description: str = "Affective Stimulus Assembly System"
    debug: bool = False
    testing: bool = False

    # Database (for local storage)
    database_url: str = "sqlite:///asa.db"

    # Logging
    log_level: str = "INFO"
    log_json: bool = False

    # Paths
    stimuli_dir: str = "assets/stimuli"
    downloads_dir: str = "downloads"

    # Affective Matching Strategy
    # Options: euclidean, manhattan, chebyshev, quadratic, mahalanobis
    affective_matching_strategy: str = "euclidean"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
