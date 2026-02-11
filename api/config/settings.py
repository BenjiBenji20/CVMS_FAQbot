from pydantic import SecretStr
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DEBUG: bool = True

    EMBEDDING_MODEL_API_KEY: SecretStr
    MODEL_NAME: str
    LLM_API_KEY: str
    LLM_NAME: str
    
    DEV_ORIGIN: str
    PROD_ORIGIN: str
    
    REQUEST_SECRET_KEY: str | None = None
    
    UPSTASH_REDIS_REST_URL: str | None = None
    UPSTASH_REDIS_REST_TOKEN: str | None = None
    UPSTASH_REDIS_PORT: int | None = 6379 # default redis port
    
    # redis conf
    def get_redis_client_uri(self) -> str:
        return (
            f"rediss://default:{self.UPSTASH_REDIS_REST_TOKEN}"
            f"@{self.UPSTASH_REDIS_REST_URL}:{self.UPSTASH_REDIS_PORT}"
        )
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
