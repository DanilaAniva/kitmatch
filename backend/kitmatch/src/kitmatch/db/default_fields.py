# Поля по умолчанию для хранения результатов детекции инструментов
detection_fields = {
    "task_id": None,  # str - идентификатор задачи
    "image_id": None,  # str - идентификатор изображения
    "original_image_path": None,  # str - путь к оригинальному изображению
    "processed_image_path": None,  # str - путь к обработанному изображению с визуализацией
    "bboxes": [],  # list of dicts - bounding boxes с координатами и классами
    "classes": [],  # list of str - уникальные классы объектов
    "scores": [],  # list of float - confidence scores для каждого bbox
    "detection_time": None,  # float - время выполнения детекции
    "errors": [],  # list of errors - ошибки при обработке
}
    