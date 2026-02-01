import numpy as np
from fastapi import UploadFile
import cv2
import base64
import binascii
from typing import Union
import io


def read_upload_image(file: Union[UploadFile, io.BytesIO]) -> np.ndarray:
    """Читает загруженное изображение (UploadFile или BytesIO) и конвертирует в RGB numpy-массив."""
    if hasattr(file, 'file'):  # Проверяем, является ли объект UploadFile
        data = file.file.read()
    else:  # В противном случае, предполагаем, что это file-like объект (например, BytesIO)
        data = file.read()
    
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def read_data_uri_image(data_uri: str) -> np.ndarray:
    """
    Преобразует data URI (data:image/...;base64,...) в RGB np.ndarray.
    """
    if not isinstance(data_uri, str) or "," not in data_uri:
        raise ValueError("Некорректный data URI")
    b64_part = data_uri.split(",", 1)[1]
    try:
        binary = base64.b64decode(b64_part, validate=True)
    except binascii.Error as e:
        raise ValueError(f"Недопустимая base64-строка: {str(e)}")
    arr = np.frombuffer(binary, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Не удалось декодировать изображение из data URI")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


