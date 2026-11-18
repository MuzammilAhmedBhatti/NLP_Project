from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MODPRECEDENT_",
        env_file=".env",
        extra="ignore",
    )

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


settings = Settings()
