from pydantic_settings import BaseSettings, SettingsConfigDict


NEON_URL_HELP = (
    "DATABASE_URL is required. Create apps/api/.env with your Neon URL, for example: "
    "DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST/DBNAME?sslmode=require"
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "AI OS API"
    environment: str = "local"
    database_url: str = ""
    clerk_jwks_url: str = ""
    clerk_issuer: str = ""
    better_auth_secret: str = "change-me"
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    access_token_expiry_minutes: int = 60
    llm_provider: str = "gemini"
    llm_model: str = "gemini-3.5-flash"
    llm_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    ollama_base_url: str = "http://localhost:11434"
    tavily_api_key: str = ""
    llm_max_output_tokens: int = 2048
    upstash_redis_rest_url: str = ""
    upstash_redis_rest_token: str = ""
    llm_cache_ttl_seconds: int = 3600
    conversation_cache_ttl_seconds: int = 300
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 768
    default_chunk_size: int = 1200
    default_chunk_overlap: int = 200
    inngest_app_id: str = "ai-os-api"
    inngest_event_key: str = ""
    inngest_signing_key: str = ""
    inngest_is_dev: bool = True
    inngest_api_base_url: str = ""
    inngest_event_api_base_url: str = ""
    frontend_origin: str = "http://localhost:3000"
    encryption_key: str = ""       # deprecated single-key fallback, still supported
    encryption_keys: str = ""      # comma-separated, newest/primary key first
    worker_poll_interval_seconds: float = 2.0
    log_level: str = "INFO"
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "ai-os"
    history_summary_trigger_messages: int = 20
    history_keep_recent_messages: int = 8
    history_summary_cache_ttl_seconds: int = 3600
    share_ttl_seconds: int = 2592000


settings = Settings()


def validate_runtime_settings() -> None:
    if not settings.database_url:
        raise RuntimeError(NEON_URL_HELP)

    if "USER:PASSWORD@HOST" in settings.database_url:
        raise RuntimeError(NEON_URL_HELP)

    if not settings.database_url.startswith(("postgresql://", "postgresql+psycopg://")):
        raise RuntimeError("DATABASE_URL must be a PostgreSQL connection string.")


def sqlalchemy_database_url() -> str:
    validate_runtime_settings()
    if settings.database_url.startswith("postgresql://"):
        return settings.database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return settings.database_url
