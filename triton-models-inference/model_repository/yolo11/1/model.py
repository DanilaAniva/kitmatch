"""
Файл: /home/ubuntu/triton-models-inference/model_repository/yolo/1/model.py

Основной смысл: Triton Inference Server backend для YOLO модели детекции объектов.
Реализует Python backend для Triton, который загружает YOLO модель и выполняет
инференс на изображениях с возвратом детекций в формате JSON.

Ключевые возможности:
- Загрузка YOLO модели из указанного пути
- Парсинг параметров конфигурации из model_config
- Обработка изображений в формате numpy arrays
- Выполнение инференса с настраиваемыми threshold'ами
- Возврат результатов в структурированном JSON формате
- Очистка ресурсов при завершении

Классы:
    TritonPythonModel: Основной класс backend для Triton Inference Server

Использование:
    # Конфигурация в config.pbtxt:
    parameters: {
        key: "model_path",
        value: {string_value: "/models/yolo.pt"}
    }
"""

import json
import numpy as np
import triton_python_backend_utils as pb_utils
from pathlib import Path
import tempfile
import torch
from ultralytics import YOLO
import ast


class TritonPythonModel:
    """
    Triton Python backend для YOLO модели детекции объектов.
    
    Основной класс, реализующий интерфейс Triton Python backend.
    Обрабатывает инициализацию модели, выполнение инференса и очистку ресурсов.
    
    Attributes:
        model_config (dict): Конфигурация модели из Triton
        conf_threshold (float): Порог уверенности для детекций
        iou_threshold (float): Порог IoU для NMS
        classes (list): Список классов для детекции
        model (YOLO): Загруженная YOLO модель
    
    Methods:
        initialize: Инициализация модели при старте Triton
        execute: Выполнение инференса на входных данных
        finalize: Очистка ресурсов при выгрузке модели
    
    Example:
        # Используется автоматически Triton сервером
        # Конфигурация через config.pbtxt
    """
    
    def initialize(self, args):
        """
        Инициализирует YOLO модель при старте Triton Inference Server.
        
        Args:
            args (dict): Аргументы от Triton, содержит model_config и другие параметры
            
        Returns:
            None
            
        Raises:
            pb_utils.TritonModelException: Если не указан обязательный параметр model_path
            
        Side Effects:
            - Загружает YOLO модель из указанного пути
            - Парсит параметры конфигурации
            - Выводит информацию о загруженной модели
            
        Configuration Parameters:
            model_path (str): Путь к YOLO модели (.pt файл)
            conf_threshold (float): Порог уверенности (по умолчанию: 0.5)
            iou_threshold (float): Порог IoU для NMS (по умолчанию: 0.45)
            classes (list): Список классов для фильтрации (по умолчанию: все)
            
        Example:
            # В config.pbtxt:
            parameters: {
                key: "model_path",
                value: {string_value: "/models/yolo.pt"}
            }
        """
        self.model_config = json.loads(args['model_config'])

        # Parse parameters from config.pbtxt
        params = self.model_config.get('parameters', {})
        model_path = params.get('model_path', {}).get('string_value')
        conf_threshold_str = params.get('conf_threshold', {}).get('string_value', "0.5")
        iou_threshold_str = params.get('iou_threshold', {}).get('string_value', "0.45")
        classes_str = params.get('classes', {}).get('string_value', "[]")

        self.conf_threshold = float(conf_threshold_str)
        self.iou_threshold = float(iou_threshold_str)
        self.classes = ast.literal_eval(classes_str)

        if not model_path:
            raise pb_utils.TritonModelException("model_path must be set in the model configuration")

        # Load YOLO model
        self.model = YOLO(model_path)

        print(f"YOLO model loaded from {model_path}")
        print(f"Confidence threshold: {self.conf_threshold}")
        print(f"IOU threshold: {self.iou_threshold}")
        print(f"Number of classes: {len(self.classes)}")

    def execute(self, requests):
        """
        Выполняет инференс YOLO модели на входных изображениях.
        
        Args:
            requests (list): Список объектов InferenceRequest от Triton
            Каждый запрос содержит тензор изображения в формате numpy array
            
        Returns:
            list: Список объектов InferenceResponse с результатами детекции
            Каждый ответ содержит JSON строку с детекциями
            
        Raises:
            pb_utils.TritonModelException: При ошибке обработки запроса
            
        Processing Pipeline:
            1. Извлечение изображения из запроса
            2. Сохранение во временный файл
            3. Выполнение инференса YOLO
            4. Обработка результатов
            5. Формирование JSON ответа
            
        Output Format:
            Список детекций в формате:
            [
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": 0.95,
                    "class": "hammer",
                    "class_id": 0
                }
            ]
            
        Example:
            # Вход: изображение 640x640x3
            # Выход: JSON с детекциями
        """
        responses = []
        for request in requests:
            image_tensor = pb_utils.get_input_tensor_by_name(request, "image")
            image_np = image_tensor.as_numpy()

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                temp_image = temp_path / "input.jpg"

                # Save image temporarily
                from PIL import Image
                img = Image.fromarray(image_np)
                img.save(temp_image)

                # Run inference
                results = self.model(
                    str(temp_image),
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )

                # Process results
                detections = []
                if len(results) > 0:
                    result = results[0]
                    boxes = result.boxes

                    if boxes is not None:
                        for i in range(len(boxes)):
                            # Get bounding box coordinates
                            bbox = boxes.xyxy[i].cpu().numpy()
                            x1, y1, x2, y2 = bbox

                            # Get confidence and class
                            conf = float(boxes.conf[i].cpu().numpy())
                            class_id = int(boxes.cls[i].cpu().numpy())
                            class_name = self.classes[class_id] if class_id < len(self.classes) else f"class_{class_id}"

                            detection = {
                                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                "confidence": conf,
                                "class": class_name,
                                "class_id": class_id
                            }
                            detections.append(detection)

            output_str = json.dumps(detections)
            output_tensor = pb_utils.Tensor("bboxes", np.array([output_str.encode('utf-8')], dtype=np.object_))
            responses.append(pb_utils.InferenceResponse([output_tensor]))

        return responses

    def finalize(self):
        """
        Очищает ресурсы при выгрузке модели из Triton Inference Server.
        
        Args:
            None
            
        Returns:
            None
            
        Side Effects:
            - Удаляет загруженную модель из памяти
            - Очищает кэш GPU
            - Освобождает связанные ресурсы
            
        Example:
            # Вызывается автоматически Triton при выгрузке модели
        """
        if hasattr(self, 'model'):
            del self.model
            torch.cuda.empty_cache()
