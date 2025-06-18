"""
Main entry point for the FastAPI application.
This script starts the FastAPI server, loads routes, and configures the server.
"""
from fastapi import FastAPI
from functools import lru_cache

from src.settings import settings
from src.nutrition.usda_client import USDAClient
from src.api.routes import router

app = FastAPI(
    title="Calorie Counter API",
    description="API for food-image nutrition estimation",
    version="1.0.0"
)
app.include_router(router)


@lru_cache                           
def get_usda_client() -> USDAClient:
    return USDAClient(settings.usda_api_key)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
