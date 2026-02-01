import numpy as np
from pymongo import MongoClient
import numpy as np
from typing import Dict, List
from kitmatch.config import Settings
from loguru import logger

class DetectionMongoClient:
    def __init__(self, host, port, username, password):
        self._client = None
        self.host, self.port = host, port
        self.username = username
        self.password = password

        settings = Settings()
        self.db_name = settings.mongodb_db_name
        self.collection_name = settings.mongodb_collection_name


    def __connect(self):
        logger.debug(f"  › [DB] Подключение к MongoDB: {self.host}:{self.port}...")
        self._client = MongoClient(
            f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}/?authSource=admin",
            serverSelectionTimeoutMS=3000,
        )
        logger.debug("  ✔ [DB] Подключение к MongoDB установлено")

    def __disconnect(self):
        if self._client is not None:
            self._client.close()
            logger.debug("  • [DB] Соединение с MongoDB закрыто")
        self._client = None

    def __enter__(self):
        self.__connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__disconnect()

    def create_detection_result(self, task_id: str, image_id: str, data: dict):
        db = self._client[self.db_name]
        doc = db[self.collection_name]
        query = {"task_id": task_id, "image_id": image_id}
        data_to_add = {"$set": {}}
        for key, value in data.items():
            key = str(key)
            data_to_add["$set"][key] = value

        logger.bind(task_id=task_id, image_id=image_id).info(f"  › [DB] Создание/обновление записи в '{self.db_name}.{self.collection_name}'")
        result = doc.update_one(query, data_to_add, upsert=True)
        return result

    def set_detection_info(self, task_id: str, image_id: str, data: dict):
        db = self._client[self.db_name]
        doc = db[self.collection_name]
        query = {"task_id": task_id, "image_id": image_id}
        data_to_add = {"$set": {}}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                value = value.tolist()
            key = str(key)
            data_to_add["$set"][key] = value

        logger.bind(task_id=task_id, image_id=image_id).info(f"  › [DB] Обновление информации для записи в '{self.db_name}.{self.collection_name}'")
        result = doc.update_one(query, data_to_add, upsert=True)
        return result

    def get_detection_info(self, task_id: str, image_id: str) -> dict:
        db = self._client[self.db_name]
        doc = db[self.collection_name]
        query = {"task_id": task_id, "image_id": image_id}
        detection_info = doc.find_one(query)
        return detection_info if detection_info else {}

    def get_detection_info_by_task_id(self, task_id: str) -> List[dict]:
        """
        Находит все записи в MongoDB, соответствующие заданному task_id.
        """
        db = self._client[self.db_name]
        doc = db[self.collection_name]
        query = {"task_id": task_id}
        # Убираем поле _id, так как оно не сериализуется в JSON
        results = list(doc.find(query, {"_id": 0}))
        return results

    def set_image_info(self, task_id: str, image_id: str, image_path: str):
        db = self._client[self.db_name]
        doc = db[self.collection_name]
        query = {"task_id": task_id, "image_id": image_id}
        data_to_add = {"$set": {}}
        data_to_add["$set"]["task_id"] = task_id
        data_to_add["$set"]["image_id"] = image_id
        data_to_add["$set"]["image_path"] = image_path
        result = doc.update_one(query, data_to_add, upsert=True)
        return result

    def get_image_info(self, task_id: str, image_id: str) -> dict:
        db = self._client[self.db_name]
        doc = db[self.collection_name]
        query = {"task_id": task_id, "image_id": image_id}
        image_info = doc.find_one(query)
        return image_info if image_info else {}
    
    def get_detection_images_paths(self, task_id: str) -> Dict[str, str]:
        db = self._client[self.db_name]
        doc = db[self.collection_name]
        query = {"task_id": task_id}
        image_info = doc.find(query)
        res = {}
        if image_info is not None:
            for info in image_info:
                res[info["image_id"]] = info["image_path"]
            return res if res else {}
        return {}

    def delete_image(self, task_id: str, image_id: str) -> bool:
        db = self._client[self.db_name]
        doc = db[self.collection_name]
        query = {"task_id": task_id, "image_id": image_id}
        result = doc.delete_one(query)
        return result.deleted_count > 0

    def get_all_task_ids(self):
        db = self._client[self.db_name]
        doc = db[self.collection_name]
        task_ids = doc.distinct("task_id")
        return task_ids

    def get_all_images_for_task(self, task_id: str) -> List[str]:
        db = self._client[self.db_name]
        doc = db[self.collection_name]
        query = {"task_id": task_id}
        images = doc.find(query, {"image_id": 1})  

        image_ids = [image["image_id"] for image in images]
        return image_ids

if __name__ == "__main__":
    import uuid

    db = DetectionMongoClient("10.5.0.117", 27017, "root", "example")
    with db:
        # banner_id = "test_banner_id_2"
        # username = "test_user"
        # task_id = "test_task_id"
        # print(db.get_banner_info("test_user", "a2784fde-e0e8-44a1-ae72-d0e1f85bc3e3", "2154ea94-ddce-4179-9dc9-2d9fe7e1488d"))
        # db.set_image_info(task_id, banner_id, "1", "bla/bla1")
        # db.set_image_info(task_id, banner_id, "2", "bla/bla2")
        # print(db.get_image_info("a2784fde-e0e8-44a1-ae72-d0e1f85bc3e3", "2154ea94-ddce-4179-9dc9-2d9fe7e1488d", "ec7a5d8d-bfc6-46ba-b953-36e143658a99"))
        # print(db.get_banner_images_paths(task_id, banner_id))
        # print(db.get_all_task_ids())
        print(db.get_all_images_for_task("test_user"))