from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MODPRECEDENT_",
        env_file=".env",
        extra="ignore",
    )

    llm_backend: str = "transformers"
    llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    chat_backend: str = "transformers"
    chat_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    api_key: str = ""
    api_base_url: str = "https://api.openai.com/v1"
    anthropic_model: str = "claude-haiku-4-5-20251001"
    anthropic_max_tokens: int = 320
    claude_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("CLAUDE_API_KEY", "ANTHROPIC_API_KEY"),
    )
    model_cache_dir: str = ""
    vision_enabled: bool = True
    vision_model: str = "Salesforce/blip-image-captioning-base"
    fallback_model: str = "Qwen/Qwen2.5-0.5B-Instruct"


settings = Settings()
