from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_chat_model: str = "qwen2.5:7b"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_timeout_sec: int = 120
    ingest_batch_size: int = 32
    vector_db_dir: str = "./vector_db"
    knowledge_base_dir: str = "./knowledge_base"
    log_level: str = "INFO"
    app_env: str = "dev"
    api_key: str = "change-me-in-env"
    enable_auth: bool = True
    cors_allow_origins: str = "*"
    rag_fast_mode: bool = True
    rag_relevance_max_chars: int = 2500
    rag_answer_max_chars: int = 4500

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def vector_db_path(self) -> Path:
        path = Path(self.vector_db_dir).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def knowledge_base_path(self) -> Path:
        path = Path(self.knowledge_base_dir).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
