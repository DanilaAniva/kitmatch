import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import os
import uuid

from kitmatch.db import detection_fields
from kitmatch.db import DetectionMongoClient, SecondDbClient
from kitmatch.config import Settings

class DetectionStorage:
    """
    Хранилище результатов детекции инструментов
    """

    def __init__(self):
        settings = Settings()
        self.detection_mongo = DetectionMongoClient(
            settings.mongodb_uri, 
            settings.mongodb_port,
            settings.mongodb_username,
            settings.mongodb_password
        )
        self.second_db = SecondDbClient()

    def create_detection_result(self, task_id: str, image_id: str):
        with self.detection_mongo:
            data = detection_fields.copy()
            data.update({"task_id": task_id, "image_id": image_id})
            self.detection_mongo.create_detection_result(task_id, image_id, data)
            return True

    def set_detection_error(self, task_id: str, image_id: str, error: str):
        with self.detection_mongo:
            data = {"errors": [error] if error else []}
            self.detection_mongo.set_detection_info(task_id, image_id, data)
            return True

    def get_detection_errors(self, task_id: str, image_id: str) -> List[str]:
        with self.detection_mongo:
            detection_info = self.detection_mongo.get_detection_info(task_id, image_id)
            return detection_info.get("errors", [])

    def _set_info(self, task_id: str, image_id: str, data):
        with self.detection_mongo:
            self.detection_mongo.set_detection_info(task_id, image_id, data)
            return True
    
    def _set_image(self, task_id: str, image_id: str, image_path: str):
        with self.detection_mongo:
            self.detection_mongo.set_image_info(task_id, image_id, image_path)
            return True

    def _set_original_image(self, task_id: str, image_id: str, original_image: np.ndarray, **kwargs):
        original_image_path = self.second_db.save_pickle(task_id, image_id, original_image, name="original_image")
        data = {
            "original_image_path": original_image_path,
        }
        return self._set_info(task_id, image_id, data)

    def get_original_image(self, task_id: str, image_id: str) -> np.ndarray:
        with self.detection_mongo:
            info = self.detection_mongo.get_detection_info(task_id, image_id)
        path = info.get("original_image_path")
        if path:
            return self.second_db.get_from_pickle(task_id, image_id, path)
        return None

    def _set_processed_image(self, task_id: str, image_id: str, processed_image: np.ndarray, **kwargs):
        processed_image_path = self.second_db.save_pickle(task_id, image_id, processed_image, name="processed_image")
        data = {
            "processed_image_path": processed_image_path,
        }
        return self._set_info(task_id, image_id, data)

    def get_processed_image(self, task_id: str, image_id: str) -> np.ndarray:
        with self.detection_mongo:
            info = self.detection_mongo.get_detection_info(task_id, image_id)
        path = info.get("processed_image_path")
        if path:
            return self.second_db.get_from_pickle(task_id, image_id, path)
        return None

    def _set_detection_results(
        self, 
        task_id: str, 
        image_id: str, 
        bboxes: List[Dict], 
        classes: List[str], 
        scores: List[float],
        detection_time: float,
        **kwargs
    ):
        data = {
            "bboxes": bboxes,
            "classes": classes,
            "scores": scores,
            "detection_time": detection_time,
        }
        return self._set_info(task_id, image_id, data)

    def get_detection_results(self, task_id: str, image_id: str) -> Dict:
        with self.detection_mongo:
            detection_info = self.detection_mongo.get_detection_info(task_id, image_id)
            return {
                "bboxes": detection_info.get("bboxes", []),
                "classes": detection_info.get("classes", []),
                "scores": detection_info.get("scores", []),
                "detection_time": detection_info.get("detection_time", 0.0),
            }

    def _set_noop(self, *args, **kwargs):
        return True

    def get_all_task_ids(self) -> List[str]:
        with self.detection_mongo:
            return self.detection_mongo.get_all_task_ids()

    def get_all_images_for_task(self, task_id: str) -> List[str]:
        with self.detection_mongo:
            return self.detection_mongo.get_all_images_for_task(task_id)

    def get_results_by_task_id(self, task_id: str) -> List[dict]:
        with self.detection_mongo:
            return self.detection_mongo.get_detection_info_by_task_id(task_id)
