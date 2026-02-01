"""
FastAPI Application Entry Point

Bootstraps the FastAPI app, logging, middleware, and routes.
"""

import threading
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from kitmatch.utils.logger import configure_logging
from kitmatch.utils.logging_middleware import RequestLoggingMiddleware
from kitmatch.routes.api import router as api_router
from kitmatch.config import Settings

# Импортируем компоненты для работы с очередями
from kitmatch.queue_utils.distributor import ModelsInferenceDistributor, PostProcessingDistributor

# Configure logging early
configure_logging(app_name="kitmatch", service="api")

settings = Settings()  # reads from environment / .env

# Глобальные переменные для дистрибьюторов
models_distributor = None
postproc_distributor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global models_distributor, postproc_distributor
    
    # Запускаем дистрибьюторы очередей в отдельных потоках
    try:
        models_distributor = ModelsInferenceDistributor(1)
        postproc_distributor = PostProcessingDistributor(1)
        
        # Запускаем дистрибьюторы в отдельных потоках
        models_thread = threading.Thread(target=models_distributor.start, daemon=True)
        postproc_thread = threading.Thread(target=postproc_distributor.start, daemon=True)
        
        models_thread.start()
        postproc_thread.start()
        
        logging.info("✔ Queue distributors started successfully")
    except Exception as e:
        logging.error(f"Failed to start queue distributors: {e}")
    
    yield
    
    # Останавливаем дистрибьюторы при завершении работы
    if models_distributor:
        try:
            models_distributor.stop()
            logging.info("✔ Models distributor stopped")
        except Exception as e:
            logging.error(f"Error stopping models distributor: {e}")
    
    if postproc_distributor:
        try:
            postproc_distributor.stop()
            logging.info("✔ Post-processing distributor stopped")
        except Exception as e:
            logging.error(f"Error stopping post-processing distributor: {e}")

app = FastAPI(
    title="kitmatch app",
    description="An API for analyzing workers inventory and match kits",
    version="1.0.0",
    lifespan=lifespan,
)

# Middleware
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(api_router, prefix="/api")

# Static files
import os
os.makedirs(settings.static_dir, exist_ok=True)
app.mount(settings.static_url, StaticFiles(directory=settings.static_dir), name="static")

@app.get("/")
def root():
    return {"status": "ok", "service": "kitmatch", "env": settings.app_env}

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
    )
