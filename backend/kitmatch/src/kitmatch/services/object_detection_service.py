import io, os, uuid, zipfile
import time
import asyncio
from typing import List, Dict, Any
from fastapi import UploadFile
from loguru import logger

from kitmatch.executors.object_finder import ObjectFinder
from kitmatch.utils.images import read_upload_image, read_data_uri_image
from kitmatch.utils.visualization import draw_bboxes, save_image
from kitmatch.utils.csv import to_csv_bytes
from kitmatch.queue_utils.send import send_mes_to_queue
from kitmatch.queue_utils.message import MessageType
from kitmatch.db.detection_storage import DetectionStorage


class ObjectDetectionService:
    def __init__(self, executor: ObjectFinder, static_dir: str, static_url: str):
        self.executor = executor
        self.static_dir = static_dir
        self.static_url = static_url
        os.makedirs(os.path.join(static_dir, "overlays"), exist_ok=True)
        self.db_storage = DetectionStorage()

    async def process_archive(self, file: UploadFile, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Обрабатывает zip-архив с изображениями, дожидается результатов
        и возвращает их.
        """
        task_id = str(uuid.uuid4())
        image_ids = []
        
        ctx_logger = logger.bind(task_id=task_id)
        ctx_logger.info(f"► [Service] Начало обработки архива: {file.filename}")

        # Читаем архив и отправляем изображения в очередь
        try:
            zip_bytes = await file.read()
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                image_files = [
                    f for f in zf.namelist() 
                    if not f.startswith('__MACOSX/') and not f.endswith('/') 
                    and f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]
                
                ctx_logger.info(f"  › [Service] В архиве найдено {len(image_files)} изображений")

                for filename in image_files:
                    image_id = str(uuid.uuid4())
                    image_ids.append(image_id)
                    
                    image_bytes = zf.read(filename)
                    image = read_upload_image(io.BytesIO(image_bytes))

                    send_mes_to_queue(
                        task_id=task_id,
                        image_id=image_id,
                        data={"image": image, "original_filename": os.path.basename(filename), "confidence_threshold": confidence_threshold},
                        msg_type=MessageType.object_detection
                    )
        except zipfile.BadZipFile:
            ctx_logger.error("  ! [Service] Ошибка: переданный файл не является zip-архивом.")
            return {"error": "Bad Zip File"}
        
        ctx_logger.info("✔ [Service] Все изображения из архива отправлены в очередь. Ожидание результатов...")

        # Ожидаем результаты из MongoDB
        start_time = time.time()
        timeout = 900  # 15 минут
        polling_interval = 2 # 2 секунды

        while time.time() - start_time < timeout:
            results = self.db_storage.get_results_by_task_id(task_id)
            if len(results) == len(image_ids):
                ctx_logger.info(f"✔ [Service] Все {len(image_ids)} результатов собраны.")
                return {"status": "complete", "results": results}
            
            await asyncio.sleep(polling_interval)

        # Если вышли по таймауту
        partial_results = self.db_storage.get_results_by_task_id(task_id)
        ctx_logger.warning(f"  ! [Service] Истекло время ожидания. Собрано {len(partial_results)} из {len(image_ids)} результатов.")
        return {
            "status": "timeout", 
            "message": f"Timeout after {timeout} seconds. Collected {len(partial_results)} of {len(image_ids)} results.",
            "results": partial_results
        }

    async def process_single_image(self, file: UploadFile, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        # Читаем изображение
        image = read_upload_image(file)
        
        # Генерируем ID для задачи и изображения
        task_id = str(uuid.uuid4())
        image_id = str(uuid.uuid4())
        
        # Отправляем изображение в очередь для обработки
        start_time = time.time()
        result = send_mes_to_queue(
            task_id=task_id,
            image_id=image_id,
            data={"image": image, "confidence_threshold": confidence_threshold},
            msg_type=MessageType.object_detection
        )
        detection_time = time.time() - start_time
        
        # Получаем результаты детекции
        bboxes = result.data.get("bboxes", [])
        classes = sorted({b["class"] for b in bboxes}) if bboxes else []
        scores = [b["confidence"] for b in bboxes] if bboxes else []
        
        # Создаем визуализацию
        overlay = draw_bboxes(image, bboxes)
        name = f"{uuid.uuid4()}.jpg"
        path = os.path.join(self.static_dir, "overlays", name)
        save_image(path, overlay)
        
        return {
            "task_id": task_id,
            "image_id": image_id,
            "image": file.filename,
            "classes": classes,
            "scores": scores,
            "bboxes": bboxes,
            "detection_time": detection_time,
            "overlay": f"{self.static_url}/overlays/{name}",
        }

    async def process_single_image_data_uri(self, data_uri: str, original_name: str | None = None) -> Dict[str, Any]:
        # Читаем изображение из data URI
        image = read_data_uri_image(data_uri)
        
        # Генерируем ID для задачи и изображения
        task_id = str(uuid.uuid4())
        image_id = str(uuid.uuid4())
        
        ctx_logger = logger.bind(task_id=task_id, image_id=image_id)
        ctx_logger.info("► [Service] Начинается обработка изображения")

        # Отправляем изображение в очередь для обработки
        ctx_logger.info("  › [Service] Отправка задачи в очередь 'models'")
        start_time = time.time()
        result = send_mes_to_queue(
            task_id=task_id,
            image_id=image_id,
            data={"image": image},
            msg_type=MessageType.object_detection
        )
        detection_time = time.time() - start_time
        ctx_logger.info(f"  ✔ [Service] Ответ из очереди получен за {detection_time:.4f} сек.")
        
        # Получаем результаты детекции
        bboxes = result.data.get("bboxes", [])
        classes = sorted({b["class"] for b in bboxes}) if bboxes else []
        scores = [b["confidence"] for b in bboxes] if bboxes else []
        
        # Создаем визуализацию
        ctx_logger.info("  › [Service] Создание overlay-изображения")
        overlay = draw_bboxes(image, bboxes)
        name = f"{uuid.uuid4()}.jpg"
        path = os.path.join(self.static_dir, "overlays", name)
        save_image(path, overlay)
        ctx_logger.info(f"  ✔ [Service] Overlay сохранен в {path}")
        
        final_response = {
            "task_id": task_id,
            "image_id": image_id,
            "classes": classes,
            "scores": scores,
            "bboxes": bboxes,
            "detection_time": detection_time,
            "overlay": f"{self.static_url}/overlays/{name}",
        }
        
        ctx_logger.info("✔ [Service] Обработка изображения успешно завершена")
        return final_response

    async def process_batch(self, files: List[UploadFile]) -> bytes:
        rows: List[Dict[str, Any]] = []

        def process_image_bytes(name: str, data: bytes):
            from numpy import frombuffer, uint8
            import cv2

            arr = frombuffer(data, dtype=uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Генерируем ID для задачи и изображения
            task_id = str(uuid.uuid4())
            image_id = str(uuid.uuid4())
            
            # Отправляем изображение в очередь для обработки
            result = send_mes_to_queue(
                task_id=task_id,
                image_id=image_id,
                data={"image": img},
                msg_type=MessageType.object_detection
            )
            
            # Получаем результаты детекции
            bboxes = result.data.get("bboxes", [])
            for b in bboxes:
                rows.append({
                    "image": name,
                    "class": b["class"],
                    "confidence": b["confidence"],
                    "x_min": b["x_min"],
                    "y_min": b["y_min"],
                    "x_max": b["x_max"],
                    "y_max": b["y_max"],
                })

        for f in files:
            filename = (f.filename or "").lower()
            data = f.file.read()
            if filename.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    for zi in zf.infolist():
                        if zi.is_dir():
                            continue
                        if not (zi.filename.lower().endswith(".jpg") or zi.filename.lower().endswith(".jpeg") or zi.filename.lower().endswith(".png")):
                            continue
                        img_bytes = zf.read(zi)
                        process_image_bytes(os.path.basename(zi.filename), img_bytes)
            else:
                process_image_bytes(os.path.basename(f.filename), data)

        return to_csv_bytes(rows)


