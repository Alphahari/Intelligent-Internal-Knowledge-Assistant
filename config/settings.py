from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OLLAMA_MODEL: str = "deepseek-r1"

settings = Settings()