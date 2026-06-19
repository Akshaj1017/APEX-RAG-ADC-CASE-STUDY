"""Central configuration. Reads from environment / .env via pydantic-settings."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mistral_api_key: str = ""
    chat_model: str = "mistral-small-latest"
    embed_model: str = "mistral-embed"

    # Provider switch. "auto" uses Mistral when a key is present, else the local
    # fake -> so the project runs offline (tests/CI) AND for real with one env var.
    embed_provider: str = "auto"       # auto | mistral | fake
    llm_provider: str = "auto"         # auto | mistral | fake
    embed_dim: int = 384               # dimension of the local fake embedder

    # RAG knobs
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5
    history_turns: int = 5

    sqlite_path: str = "apex.db"
    chroma_path: str = "chroma_store"


settings = Settings()
