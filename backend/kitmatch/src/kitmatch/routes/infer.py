import io
import os
import json
from fastapi import APIRouter, UploadFile, File, Depends, Form
from fastapi.responses import StreamingResponse, Response

from kitmatch.services.object_detection_service import ObjectDetectionService
from kitmatch.config import Settings


router = APIRouter(prefix="/v1/infer", tags=["infer"])


def get_service() -> ObjectDetectionService:
    from kitmatch.executors.object_finder import ObjectFinder

    settings = Settings()
    executor = ObjectFinder(host=settings.triton_url, service_name=settings.triton_model_name)
    static_dir = getattr(settings, "static_dir", "static")
    static_url = getattr(settings, "static_url", "/static")
    return ObjectDetectionService(executor, static_dir=static_dir, static_url=static_url)


@router.post("/image")
async def infer_image(
    file: UploadFile = File(...), 
    confidence_threshold: float = Form(0.5),
    svc: ObjectDetectionService = Depends(get_service)
):
    return await svc.process_single_image(file, confidence_threshold)


@router.post("/archive")
async def infer_archive(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5),
    svc: ObjectDetectionService = Depends(get_service)
):
    """
    Принимает zip-архив, обрабатывает все изображения и дожидается
    полного результата. Возвращает json-файл с результатами.
    """
    if not file.filename or not file.filename.lower().endswith('.zip'):
        return Response(content=json.dumps({"error": "File is not a zip archive"}), media_type="application/json", status_code=400)
    
    processing_result = await svc.process_archive(file, confidence_threshold)

    archive_name = os.path.splitext(file.filename)[0]
    json_filename = f"results_{archive_name}.json"
    
    return Response(
        content=json.dumps(processing_result, indent=2, ensure_ascii=False),
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="{json_filename}"'
        }
    )


@router.post("/batch")
async def infer_batch(files: list[UploadFile] = File(...), svc: ObjectDetectionService = Depends(get_service)):
    csv_bytes = await svc.process_batch(files)
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="detections.csv"'},
    )


