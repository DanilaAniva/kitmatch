import os
import numpy as np
import time
from typing import List, Tuple, Dict, Any
from urllib.parse import urlparse
from loguru import logger

from kitmatch.utils.registry import register_executor
from kitmatch.executors.base import BaseExecutor
from kitmatch.queue_utils import MessageType
from kitmatch.executors.triton_client import infer_yolo_object_detector


@register_executor(message_type=MessageType.object_detection, triton_service_name="yolo11")
class ObjectFinder(BaseExecutor):
    """
    Экзекутор для детекции объектов через Triton с использованием модели YOLO.
    """

    def __init__(self, host: str, service_name: str | None = None, mock: bool | None = None):
        super().__init__(host, service_name)
        
        parsed_url = urlparse(self.host)
        self.triton_host = parsed_url.hostname
        self.triton_port = parsed_url.port
        
        if not self.service_name:
            raise ValueError("triton_service_name не задан ни декоратором, ни в конструкторе")
        if not self.triton_host or not self.triton_port:
            raise ValueError(f"Некорректный URL Triton: {self.host}. Ожидается формат 'http://<host>:<port>'")

    def get_model_result(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Вызывает Triton с использованием HTTP-клиента и возвращает «сырые» результаты модели.
        """
        logger.info(f"  › [Executor] Вызов модели '{self.service_name}' на Triton сервере: {self.triton_host}:{self.triton_port}")
        start_time = time.time()
        
        try:
            raw_result = infer_yolo_object_detector(
                host=self.triton_host,
                port=self.triton_port,
                model_name=self.service_name,
                image=image
            )
        except Exception as e:
            logger.error(f"  ! [Executor] Ошибка при вызове Triton: {e}")
            return []

        inference_time = time.time() - start_time
        logger.info(f"  ✔ [Executor] Ответ от Triton получен за {inference_time:.4f} сек.")
        return raw_result

    def model_inference(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Выполняет инференс модели на переданных данных.
        """
        image = data.get("image")
        if image is None:
            raise ValueError("Image data is missing")
        
        # Результат инференса — это и есть bounding boxes
        bboxes = self.get_model_result(image)
        return {"bboxes": bboxes}

    def get_main_result(self, data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("  › [Executor] Пост-обработка результатов модели")
        
        # Данные уже в нужном формате после model_inference
        bboxes = data.get("bboxes", [])
        confidence_threshold = data.get("confidence_threshold", 0.5) # Порог по умолчанию 0.5
        
        if not bboxes:
            logger.warning("  ! [Executor] Модель не вернула bounding boxes для пост-обработки")
            return {"bboxes": []}
        
        processed_bboxes = self._postprocess_bboxes(bboxes)
        
        # Фильтруем bboxes по порогу уверенности
        filtered_bboxes = [
            box for box in processed_bboxes 
            if box.get("confidence", 0.0) >= confidence_threshold
        ]

        logger.info(f"  ✔ [Executor] Пост-обработка завершена, найдено {len(filtered_bboxes)} объектов (порог: {confidence_threshold})")
        return {"bboxes": filtered_bboxes}

    @staticmethod
    def _postprocess_bboxes(raw_boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Преобразует bbox из [x1, y1, x2, y2] в формат с x_min, y_min и т.д.
        """
        results: List[Dict[str, Any]] = []
        for box in raw_boxes:
            bbox_coords = box.get("bbox", [])
            if len(bbox_coords) < 4:
                continue

            results.append({
                "class": box.get("class", "unknown"),
                "confidence": float(box.get("confidence", 0.0)),
                "x_min": float(bbox_coords[0]),
                "y_min": float(bbox_coords[1]),
                "x_max": float(bbox_coords[2]),
                "y_max": float(bbox_coords[3]),
            })
        return results
