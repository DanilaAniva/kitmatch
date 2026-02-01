"""
Создание bbox-кропов инструментов с использованием GroundingDINO

Этот скрипт автоматически создает обрезанные изображения инструментов, используя
детекцию через GroundingDINO. Предназначен для подготовки данных для обучения
классификаторов на чистых изображениях инструментов без фонового шума.

Основная логика:
1. Сканирует папку класса с исходными изображениями инструментов
2. Для каждого изображения запускает GroundingDINO детекцию по промпту "tool"
3. Выбирает лучший (по confidence) bounding box 
4. Расширяет bbox на заданный процент для захвата контекста
5. Обрезает изображение по расширенному bbox
6. При необходимости поворачивает в вертикальную ориентацию
7. Сохраняет результат в подпапку "cropped"

Модели:
- GroundingDINO (iSEE-Laboratory/llmdet_base по умолчанию) для детекции bbox
- Поддерживает различные модели через HuggingFace transformers
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from tqdm import tqdm

# Используем подход из process_folders_crops_y.py: GroundingDinoProcessor + MMGroundingDinoForObjectDetection
from transformers import GroundingDinoProcessor
from transformers.models.mm_grounding_dino import MMGroundingDinoForObjectDetection


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}


def select_device() -> torch.device:
    """
    Выбирает оптимальное устройство для вычислений.
    
    Приоритет: CUDA > MPS (Apple Silicon) > CPU
    
    Returns:
        Устройство PyTorch
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def list_source_images(class_dir: Path) -> List[Path]:
    """
    Находит все исходные изображения в папке класса.
    
    Args:
        class_dir: Путь к папке класса
        
    Returns:
        Отсортированный список Path объектов
    """
    files = []
    for p in sorted(class_dir.iterdir()):
        if p.is_file() and p.suffix in IMAGE_EXTENSIONS:
            files.append(p)
    return files


def ensure_dir(path: Path) -> None:
    """
    Создаёт папку, если она не существует.
    
    Args:
        path: Путь к папке
    """
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def clamp(val: float, lo: float, hi: float) -> float:
    """
    Ограничивает значение в заданном диапазоне.
    
    Args:
        val: Исходное значение
        lo: Минимальное значение
        hi: Максимальное значение
        
    Returns:
        Ограниченное значение
    """
    return max(lo, min(hi, val))


def expand_box(xmin: float, ymin: float, xmax: float, ymax: float, w: int, h: int, ratio: float = 0.05) -> Tuple[int, int, int, int]:
    """
    Расширяет bounding box на заданный коэффициент в каждую сторону.
    
    Расширяет относительно центра bbox и ограничивает результат
    границами изображения.
    
    Args:
        xmin, ymin, xmax, ymax: Координаты исходного bbox
        w, h: Размеры изображения
        ratio: Коэффициент расширения на каждую сторону
        
    Returns:
        Координаты расширенного bbox (x1, y1, x2, y2)
    """
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    bw = (xmax - xmin)
    bh = (ymax - ymin)
    bw *= (1.0 + ratio * 2.0)
    bh *= (1.0 + ratio * 2.0)
    x1 = int(clamp(cx - bw / 2.0, 0, w - 1))
    y1 = int(clamp(cy - bh / 2.0, 0, h - 1))
    x2 = int(clamp(cx + bw / 2.0, 0, w))
    y2 = int(clamp(cy + bh / 2.0, 0, h))
    if x2 <= x1: x2 = min(w, x1 + 1)
    if y2 <= y1: y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


@torch.inference_mode()
def detect_top1_box(img: Image.Image, model: MMGroundingDinoForObjectDetection, processor: GroundingDinoProcessor, device: torch.device, threshold: float = 0.1) -> Tuple[int, int, int, int] | None:
    """
    Находит лучший bounding box с помощью GroundingDINO.
    
    Запускает детекцию по промпту "tool" и выбирает bbox с максимальным confidence.
    
    Args:
        img: Изображение PIL в RGB формате
        model: Модель GroundingDINO
        processor: Процессор GroundingDINO
        device: Устройство для вычислений
        threshold: Минимальный порог для детекции
        
    Returns:
        Координаты лучшего bbox (xmin, ymin, xmax, ymax) или None, если ничего не найдено
    """
    inputs = processor(text=[["tool"]], images=img, return_tensors="pt")
    inputs = inputs.to(device)
    outputs = model(**inputs)
    target_sizes = torch.tensor([img.size[::-1]], device=device)
    results = processor.post_process_grounded_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=threshold
    )
    boxes = results[0].get("boxes")
    scores = results[0].get("scores")
    if boxes is None or scores is None or scores.numel() == 0:
        return None
    idx = int(torch.argmax(scores).item())
    xmin, ymin, xmax, ymax = [float(v) for v in boxes[idx].tolist()]
    return int(xmin), int(ymin), int(xmax), int(ymax)


def main() -> None:
    """
    Главная функция для создания bbox-кропов для одного класса.
    
    Обрабатывает все изображения в папке класса, создаёт обрезанные версии.
    
    Примеры запуска:
    # Обработка одного класса
    python make_bbox_crops.py --class_dir "/path/to/dataset/Отвертка «-»"
    
    # Ограничение количества кропов до 50
    python make_bbox_crops.py --class_dir "/path/to/class" --max_n 50
    
    # Использование другой модели и порогов
    python make_bbox_crops.py --class_dir "/path/to/class" --model_name "IDEA-Research/grounding-dino-base" --threshold 0.15
    
    # Увеличенное расширение bbox на 10%
    python make_bbox_crops.py --class_dir "/path/to/class" --expand 0.10
    """
    parser = argparse.ArgumentParser(description="Generate bbox-based crops for a single class directory")
    parser.add_argument("--class_dir", type=str, required=True, help="Path to class directory (contains original images)")
    parser.add_argument("--max_n", type=int, default=100, help="Max number of new crops to create")
    parser.add_argument("--model_name", type=str, default="iSEE-Laboratory/llmdet_base", help="GroundingDINO HF model id")
    parser.add_argument("--threshold", type=float, default=0.10, help="Detection threshold")
    parser.add_argument("--expand", type=float, default=0.06, help="Relative bbox expansion on each side")
    args = parser.parse_args()

    class_dir = Path(args.class_dir)
    assert class_dir.exists() and class_dir.is_dir(), f"Class dir not found: {class_dir}"

    cropped_dir = class_dir / "cropped"
    ensure_dir(cropped_dir)

    # Device + model
    device = select_device()
    processor = GroundingDinoProcessor.from_pretrained(args.model_name)
    model = MMGroundingDinoForObjectDetection.from_pretrained(args.model_name, low_cpu_mem_usage=True).to(device).eval()

    src_images = list_source_images(class_dir)
    created = 0
    total_candidates = len(src_images)

    pbar = tqdm(src_images, desc="Crops", unit="img")
    for img_path in pbar:
        if created >= args.max_n:
            break
        stem, ext = img_path.stem, img_path.suffix
        out_path = cropped_dir / f"{stem}_cropped{ext}"
        if out_path.exists():
            continue
        try:
            with Image.open(img_path).convert("RGB") as im:
                box = detect_top1_box(im, model, processor, device, threshold=args.threshold)
                if box is None:
                    continue
                xmin, ymin, xmax, ymax = box
                x1, y1, x2, y2 = expand_box(xmin, ymin, xmax, ymax, im.width, im.height, ratio=args.expand)
                crop = im.crop((x1, y1, x2, y2))
                # Привести к вертикальной ориентации, если нужно
                if crop.width > crop.height:
                    crop = crop.rotate(90, expand=True)
                crop.save(out_path)
                created += 1
                pbar.set_postfix({"created": created})
        except Exception:
            # Тихо пропускаем проблемные файлы
            continue

    print(f"Готово. Создано: {created} кропов (лимит {args.max_n}), обработано: {total_candidates}")


if __name__ == "__main__":
    main()


