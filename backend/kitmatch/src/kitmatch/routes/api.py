"""
API Routes

This module defines the FastAPI routes
It handles incoming HTTP requests, validates them using Pydantic models,
delegates the core business logic to the appropriate services, and returns
the results as JSON responses.
"""

import logging
import sys
import uuid
from fastapi import APIRouter, Depends, HTTPException, Request, Body
from kitmatch.services.object_detection_service import ObjectDetectionService
from kitmatch.routes.infer import router as infer_router
import binascii
from kitmatch.utils.logger import logger
from kitmatch.models.detection_models import DetectionRequest, DetectionResponse

# Настройка логирования
root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.handlers.clear()
root.addHandler(handler)

logging.getLogger("pika").setLevel(logging.WARNING)
logging.getLogger("pymongo").setLevel(logging.WARNING)
logging.info("Logging is set!")

def get_service() -> ObjectDetectionService:
    from kitmatch.executors.object_finder import ObjectFinder
    from kitmatch.config import Settings

    settings = Settings()
    executor = ObjectFinder(host=settings.triton_url, service_name=settings.triton_model_name)
    return ObjectDetectionService(executor, static_dir=settings.static_dir, static_url=settings.static_url)

router = APIRouter()

@router.get("/ping")
def ping():
    logger.info("Ping endpoint called")
    return {"message": "pong"}

router.include_router(infer_router)

@router.post("/analyze", response_model=DetectionResponse)
async def analyze(
    request: DetectionRequest, 
    svc: ObjectDetectionService = Depends(get_service)
):
    logger.info("HTTP request started")
    try:
        image_data = request.image.get("data")
        if not image_data or request.image.get("encoding") != "data_uri":
            raise ValueError("Некорректный формат изображения. Ожидается data_uri.")
        
        # Логируем начало обработки
        logger.info(f"► [Process] Starting image analysis")
        
        result = await svc.process_single_image_data_uri(image_data)
        
        # Логируем успешное завершение
        logger.info("✔ [Process] Image analysis completed successfully")
        logger.info("HTTP request completed")
        return result
    except (ValueError, binascii.Error) as e:
        logger.error(f"Ошибка обработки запроса: {str(e)}")
        raise HTTPException(status_code=400, detail={"error": str(e)})
    except Exception as e:
        logger.error(f"Непредвиденная ошибка: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": "Внутренняя ошибка сервера"})
