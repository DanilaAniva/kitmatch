import numpy as np
import cv2
from typing import List, Dict, Any
from PIL import Image, ImageDraw, ImageFont

# Набор предопределенных цветов для классов
COLORS = [
    (255, 56, 56), (255, 157, 151), (255, 112, 255), (178, 0, 255), (204, 204, 255),
    (0, 153, 255), (51, 255, 255), (0, 255, 0), (255, 255, 0), (255, 128, 0)
]

def draw_bboxes(image: np.ndarray, bboxes: List[Dict[str, Any]]) -> np.ndarray:
    """
    Рисует bounding boxes, названия классов и score на изображении,
    используя Pillow для корректного отображения кириллицы.
    """
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    class_colors = {}
    color_index = 0

    try:
        # Указываем путь к шрифту, который мы установили в Dockerfile
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except IOError:
        # Фоллбэк на дефолтный шрифт, если кастомный не найден
        font = ImageFont.load_default()

    for b in bboxes or []:
        class_name = b.get("class", "unknown")
        
        if class_name not in class_colors:
            class_colors[class_name] = COLORS[color_index % len(COLORS)]
            color_index += 1
        
        color = class_colors[class_name]
        
        p1 = (int(b.get("x_min", 0)), int(b.get("y_min", 0)))
        p2 = (int(b.get("x_max", 0)), int(b.get("y_max", 0)))
        
        # Рисуем прямоугольник
        draw.rectangle([p1, p2], outline=color, width=2)
        
        # Формируем текст метки
        score = b.get("confidence", 0.0)
        label = f"{class_name} {score:.2f}"
        
        # Рассчитываем размер текста, чтобы нарисовать фон
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Рисуем фон для текста
        label_p1 = (p1[0], p1[1] - text_height - 6)
        label_p2 = (p1[0] + text_width + 4, p1[1])
        draw.rectangle([label_p1, label_p2], fill=color)
        
        # Рисуем текст (белым цветом на фоне)
        draw.text((p1[0] + 2, p1[1] - text_height - 4), label, font=font, fill=(255, 255, 255))
        
    return np.array(img_pil)


def save_image(path: str, image: np.ndarray) -> None:
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)


