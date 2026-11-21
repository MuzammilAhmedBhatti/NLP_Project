from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MODPRECEDENT_",
        env_file=".env",
        extra="ignore",
    )

    llm_backend: str = "ollama"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


settings = Settings()
