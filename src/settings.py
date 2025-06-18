from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    usda_api_key: str = Field(..., env="USDA_API_KEY")

settings = Settings() 