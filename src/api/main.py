"""
Main entry point for the FastAPI application.
This script starts the FastAPI server, loads routes, and configures the server.
"""
from fastapi import FastAPI
from src.api.routes import router


# Initialize the FastAPI app
app = FastAPI(
    title="Nutrition Facts API",
    description="API for processing food images and returning its nutritional facts.",
    version="1.0.0")

app.include_router(router)

# Run the FastAPI application when this script is run directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
