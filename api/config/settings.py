from pydantic import SecretStr
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DEBUG: bool = True

    EMBEDDING_MODEL_API_KEY: SecretStr
    MODEL_NAME: str
    LLM_API_KEY: str
    LLM_NAME: str
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
