"""
Автоматическая аннотация COCO с использованием GroundingDINO и SAM-2

Этот скрипт выполняет автоматическую аннотацию датасета инструментов с созданием разметки 
в формате COCO JSON. Он использует две нейронные сети:
- GroundingDINO (rziga/mm_grounding_dino_large_all) для детекции bbox по текстовому промпту "tool"
- SAM-2 (hiera_large) для генерации точных масок по найденным bbox

Основная логика:
1. Сканирует папки-классы в датасете (каждая папка = один класс инструментов)
2. Для каждого изображения запускает детекцию GroundingDINO
3. Для найденных bbox генерирует маски через SAM-2
4. Сохраняет результаты: визуализации, маски и JSON-аннотации
5. Агрегирует все результаты в единый COCO JSON файл

Поддерживает многопроцессную обработку с распределением по GPU.
"""

import os
import sys
import glob
import json
import math
import time
from typing import List, Dict, Tuple, Optional
import multiprocessing as mp
import threading
import queue

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch import cuda as torch_cuda


# ==========================
# Константы/настройки по умолчанию
# ==========================

# Папка датасета, где лежат подпапки-классы (можно переопределить аргументом CLI)
DEFAULT_DATASET_DIR = \
    "/home/ubuntu/diabert/dataset/dataset"

# Исключённые подпапки-категории
EXCLUDED_FOLDERS = {
    "Групповые для тренировки",
    "Инструменты с линейкой",
}

# Имя подпапки для сохранения разметки внутри каждой папки класса
SUBDIR_OUT = "razmetka_bbox_mask"

# GroundingDINO
GROUNDING_DINO_MODEL = "rziga/mm_grounding_dino_large_all"
GROUNDING_PROMPT = "tool"
BOX_THR = 0.25
TEXT_THR = 0.25

# SAM-2
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT = "/home/ubuntu/sam2/checkpoints/sam2.1_hiera_large.pt"

# Устройства
FORCE_SINGLE_GPU = True


# ==========================
# Импорты моделей (как в исходном скрипте)
# ==========================
try:
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
except Exception:
    raise SystemExit(
        "Не найден GroundingDINO в transformers. Установите transformers >=4.40 и проверьте модель rziga/mm_grounding_dino_large_all"
    )

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except Exception:
    sam2_root = "/home/ubuntu/sam2"
    if os.path.isdir(sam2_root):
        sys.path.insert(0, sam2_root)
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    else:
        raise SystemExit(
            "SAM-2 не найден. Установите его из /home/ubuntu/sam2: pip install -e '.[notebooks]'"
        )


# ==========================
# Утилиты
# ==========================
def log_gpu_memory(tag: str) -> None:
    """
    Логирует текущее использование GPU памяти.
    
    Args:
        tag: Метка для идентификации точки измерения
    """
    if not torch_cuda.is_available():
        return
    allocated = torch_cuda.memory_allocated() / (1024 * 1024)
    reserved = torch_cuda.memory_reserved() / (1024 * 1024)
    max_alloc = torch_cuda.max_memory_allocated() / (1024 * 1024)
    print(
        f"[GPU MEM][{tag}] allocated={allocated:.0f} MB, reserved={reserved:.0f} MB, peak={max_alloc:.0f} MB"
    )
def rle_encode_mask_bool(mask: np.ndarray) -> Dict:
    """
    RLE-энкодер булевой маски в uncompressed RLE для формата COCO.

    Кодирует бинарную маску в формат Run Length Encoding (RLE). 
    Всегда начинает с количества нулей (фона), затем чередует фон/передний план.
    
    Args:
        mask: Булева маска размера [H, W]
        
    Returns:
        Словарь с полями:
        - size: [H, W] - размеры маски
        - counts: List[int] - длины участков (фон, объект, фон, объект, ...)
    """
    h, w = mask.shape
    flat = mask.reshape(-1).astype(np.uint8)
    counts: List[int] = []
    prev_val = 0
    run_len = 0
    for v in flat:
        iv = int(v)
        if iv == prev_val:
            run_len += 1
        else:
            counts.append(int(run_len))
            run_len = 1
            prev_val = iv
    counts.append(int(run_len))
    return {"size": [int(h), int(w)], "counts": counts}


def rle_area(rle: Dict) -> float:
    """
    Вычисляет площадь объекта по RLE-кодировке.
    
    Args:
        rle: RLE-кодировка с полями 'counts' (список длин участков)
        
    Returns:
        Площадь объекта в пикселях (сумма длин участков переднего плана)
    """
    counts = rle.get("counts", [])
    s = 0
    for i in range(1, len(counts), 2):
        try:
            s += int(counts[i])
        except Exception:
            pass
    return float(s)


def compute_tight_bbox_from_mask(mask: np.ndarray) -> List[float]:
    """
    Вычисляет минимальный ограничивающий прямоугольник для маски.
    
    Нормализует маску к формату 2D bool (H, W) и находит координаты bbox.
    
    Args:
        mask: Маска произвольного формата (может быть 1D, 2D, 3D, bool, int)
        
    Returns:
        Список [x1, y1, x2, y2] - координаты bbox в формате XYXY
        Возвращает [0, 0, 0, 0] если маска пустая или невалидная
    """
    # Нормализация маски к 2D bool (H, W), чтобы избежать сбоев np.where для 1D/ND
    if mask is None:
        return [0.0, 0.0, 0.0, 0.0]
    m = np.asarray(mask)
    # Приводим к булевому типу
    if m.dtype != bool:
        m = m != 0
    # Если 3D: схлопываем канальный размер логическим ИЛИ
    if m.ndim == 3:
        # Пытаемся привести к (H, W)
        # Варианты: (H, W, C) или (C, H, W)
        if m.shape[0] <= 4 and m.shape[0] != m.shape[1]:
            # Вероятно (C, H, W)
            m = m.any(axis=0)
        else:
            # Вероятно (H, W, C)
            m = m.any(axis=2)
    elif m.ndim != 2:
        # 0D/1D/ND>3 — невалидно, возвращаем пустой bbox
        return [0.0, 0.0, 0.0, 0.0]

    ys, xs = np.where(m)
    if xs.size == 0 or ys.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return [float(x1), float(y1), float(x2 + 1), float(y2 + 1)]


def save_overlay_with_masks(img: Image.Image, dets: List[Dict], out_path: str, class_name: str) -> None:
    """
    Сохраняет визуализацию изображения с наложенными масками и bbox.
    
    Создает полупрозрачный оверлей с красными масками, белыми bbox и подписями классов.
    
    Args:
        img: Исходное изображение PIL
        dets: Список детекций с полями 'mask' и 'bbox_xyxy'
        out_path: Путь для сохранения визуализации
        class_name: Имя класса для подписей
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    H, W = img.height, img.width
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.imshow(np.array(img))

    # RGBA оверлей для всех масок
    overlay = np.zeros((H, W, 4), dtype=np.uint8)
    alpha_val = int(0.35 * 255)
    color = np.array([255, 0, 0], dtype=np.uint8)
    for d in dets:
        mask = d.get("mask")
        if mask is None:
            continue
        overlay[..., 3][mask] = np.maximum(overlay[..., 3][mask], alpha_val)
        overlay[..., :3][mask] = color
    if np.any(overlay[..., 3] > 0):
        ax.imshow(overlay, interpolation="nearest")

    for d in dets:
        x1, y1, x2, y2 = d["bbox_xyxy"]
        w = max(0.0, float(x2) - float(x1))
        h = max(0.0, float(y2) - float(y1))
        rect = patches.Rectangle((x1, y1), w, h, linewidth=3, edgecolor=(1.0, 0.3, 0.0, 1.0), facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, max(0, y1 - 10), class_name, fontsize=10, color='white',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=(1.0, 0.3, 0.0, 0.95)))

    ax.set_xlim(0, img.width)
    ax.set_ylim(img.height, 0)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# ==========================
# Обёртки моделей (скопировано по логике из исходного файла)
# ==========================
class GroundingDINODetector:
    """
    Обертка для модели GroundingDINO из transformers.
    
    Выполняет zero-shot детекцию объектов по текстовому описанию.
    Возвращает bounding boxes в формате XYXY с конфиденсом.
    """
    def __init__(self, model_name: str, prompt: str, box_thr: float, text_thr: float, device: torch.device):
        """
        Инициализирует детектор GroundingDINO.
        
        Args:
            model_name: Название модели на HuggingFace Hub
            prompt: Текстовый промпт для детекции (нпр. "tool")
            box_thr: Порог для скора bbox
            text_thr: Порог для скора текста
            device: Устройство PyTorch для вычислений
        """
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(device).eval()
        self.prompt_txt = " . ".join([prompt]) + " ."
        self.box_thr = box_thr
        self.text_thr = text_thr
        self.device = device

    @torch.no_grad()
    def detect(self, image: Image.Image) -> List[Dict]:
        """
        Выполняет детекцию объектов на изображении.
        
        Args:
            image: Изображение PIL
            
        Returns:
            Список словарей с полями:
            - bbox_xyxy: [x1, y1, x2, y2] - координаты bounding box
            - score_vlm: Конфиденс детекции
        """
        inputs = self.processor(images=[image], text=[self.prompt_txt], return_tensors="pt").to(self.device)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=(self.device.type == "cuda")):
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs, input_ids=inputs.input_ids,
            threshold=self.box_thr, text_threshold=self.text_thr,
            target_sizes=[image.size[::-1]],
        )
        res = results[0]
        boxes = res.get("boxes", [])
        scores = res.get("scores", [])
        dets = []
        for i in range(min(len(boxes), len(scores))):
            bb = boxes[i].detach().float().cpu().tolist()
            sc = float(scores[i].detach().cpu())
            dets.append({"bbox_xyxy": [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])], "score_vlm": sc})
        return dets


class SAM2Segmenter:
    """
    Обертка для модели SAM-2 (сегментация).
    
    Генерирует точные маски объектов по bounding boxes.
    Поддерживает как одиночную, так и пакетную обработку.
    """
    def __init__(self, cfg: str, ckpt: str, device: torch.device):
        """
        Инициализирует SAM-2 сегментатор.
        
        Args:
            cfg: Путь к YAML конфигу SAM-2
            ckpt: Путь к чекпоинту SAM-2
            device: Устройство PyTorch
        """
        self.model = build_sam2(cfg, ckpt).to(device).eval()
        self.predictor = SAM2ImagePredictor(self.model)
        self.device = device
        self._img_np = None

    def set_image(self, img_np: np.ndarray) -> None:
        """
        Устанавливает изображение для сегментации.
        
        Args:
            img_np: Изображение в формате numpy RGB [H, W, 3]
        """
        self._img_np = img_np
        self.predictor.set_image(img_np)

    def mask_from_box(self, xyxy: List[float]) -> np.ndarray:
        """
        Генерирует маску для одного bounding box.
        
        Args:
            xyxy: Координаты [x1, y1, x2, y2] в формате XYXY
            
        Returns:
            Булева маска [H, W] - True для пикселей объекта
        """
        assert self._img_np is not None, "Сначала вызовите set_image()"
        b = np.array(xyxy, dtype=np.float32)
        masks, ious, _ = self.predictor.predict(box=b[None, :], multimask_output=True)
        # Нормализация формата выходов
        if masks is None:
            return np.zeros((self._img_np.shape[0], self._img_np.shape[1]), dtype=bool)
        m = np.asarray(masks)
        if m.size == 0:
            return np.zeros((self._img_np.shape[0], self._img_np.shape[1]), dtype=bool)
        # Возможные формы: (M, H, W) при одной коробке; иногда (H, W)
        if m.ndim == 3:
            best = 0
            try:
                best = int(np.argmax(np.asarray(ious).reshape(-1)))
            except Exception:
                best = 0
            best = max(0, min(best, m.shape[0] - 1))
            return m[best].astype(bool)
        if m.ndim == 2:
            return m.astype(bool)
        # Непредвиденный формат — возвращаем пустую маску
        return np.zeros((self._img_np.shape[0], self._img_np.shape[1]), dtype=bool)

    def masks_from_boxes_batch(self, boxes_xyxy: List[List[float]]) -> List[np.ndarray]:
        """
        Пакетное предсказание масок для нескольких bounding boxes.
        
        Оптимизированная версия обработки нескольких boxes за один вызов.
        При ошибке откатывается на последовательную обработку.
        
        Args:
            boxes_xyxy: Список bounding boxes в формате [[x1,y1,x2,y2], ...]
            
        Returns:
            Список булевых масок [H, W] для каждого box
        """
        assert self._img_np is not None, "Сначала вызовите set_image()"
        if not boxes_xyxy:
            return []
        try:
            # Попытка пакетного вызова
            boxes_np = np.array(boxes_xyxy, dtype=np.float32)
            masks, ious, _ = self.predictor.predict(box=boxes_np, multimask_output=True)
            m = np.asarray(masks)
            result_masks: List[np.ndarray] = []
            num_boxes = len(boxes_xyxy)

            # Вариант А: (B, M, H, W)
            if m.ndim == 4 and m.shape[0] == num_boxes:
                for i in range(num_boxes):
                    try:
                        best_idx = int(np.argmax(np.asarray(ious[i]).reshape(-1)))
                    except Exception:
                        best_idx = 0
                    best_idx = max(0, min(best_idx, m.shape[1] - 1))
                    result_masks.append(m[i, best_idx].astype(bool))
                return result_masks

            # Вариант Б: (B, H, W) — по одной маске на коробку
            if m.ndim == 3 and m.shape[0] == num_boxes:
                for i in range(num_boxes):
                    result_masks.append(m[i].astype(bool))
                return result_masks

            # Вариант В: (M, H, W) — один бокс, несколько масок
            if m.ndim == 3 and num_boxes == 1:
                try:
                    best_idx = int(np.argmax(np.asarray(ious).reshape(-1)))
                except Exception:
                    best_idx = 0
                best_idx = max(0, min(best_idx, m.shape[0] - 1))
                return [m[best_idx].astype(bool)]

            # Вариант Г: (H, W) — один бокс, одна маска
            if m.ndim == 2 and num_boxes == 1:
                return [m.astype(bool)]

            # Непредвиденный формат — откатываемся на по-одному
            return [self.mask_from_box(box) for box in boxes_xyxy]
        except Exception:
            # Fallback на по-одному
            return [self.mask_from_box(box) for box in boxes_xyxy]


# ==========================
# Основной процесс
# ==========================
def find_device_pair() -> Tuple[torch.device, torch.device]:
    """
    Определяет оптимальные устройства для детектора и сегментатора.
    
    По умолчанию использует один GPU для обеих моделей (FORCE_SINGLE_GPU=True).
    
    Returns:
        Кортеж (device_detector, device_segmenter)
    """
    num_gpus = torch.cuda.device_count()
    if FORCE_SINGLE_GPU and torch.cuda.is_available():
        det = torch.device("cuda:0")
        seg = torch.device("cuda:0")
        print(f"[INFO] Форсируем один GPU для всех моделей: {det}")
        return det, seg
    if num_gpus > 0:
        dev = torch.device("cuda:0")
        print(f"[INFO] Используется устройство: {dev}")
        return dev, dev
    dev = torch.device("cpu")
    print(f"[INFO] CUDA не найдена. Используется CPU")
    return dev, dev


def collect_class_folders(dataset_dir: str) -> List[str]:
    """
    Собирает список папок-классов в датасете.
    
    Исключает папки из EXCLUDED_FOLDERS.
    
    Args:
        dataset_dir: Путь к корневой папке датасета
        
    Returns:
        Отсортированный список полных путей к папкам-классам
    """
    folders = []
    for name in os.listdir(dataset_dir):
        full = os.path.join(dataset_dir, name)
        if not os.path.isdir(full):
            continue
        if name in EXCLUDED_FOLDERS:
            continue
        folders.append(full)
    folders.sort()
    return folders


def list_images(folder: str) -> List[str]:
    """
    Находит все изображения в папке.
    
    Args:
        folder: Путь к папке
        
    Returns:
        Отсортированный список путей к файлам изображений
    """
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.WEBP"]
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(folder, pat)))
    paths.sort()
    return paths


def process_image_with_models(img_path: str, detector: GroundingDINODetector, segmenter: SAM2Segmenter) -> List[Dict]:
    """
    Обрабатывает одно изображение полным пайплайном: детекция + сегментация.
    
    Args:
        img_path: Путь к файлу изображения
        detector: Инициализированный GroundingDINO детектор
        segmenter: Инициализированный SAM-2 сегментатор
        
    Returns:
        Список детекций с полями:
        - bbox_xyxy: Координаты tight bbox для маски
        - mask: Булева маска [H, W]
        - score_vlm: Конфиденс от GroundingDINO
        - _timing_*: Метрики времени для первого изображения
    """
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"[WARN] Пропуск (не открыть): {img_path}, {e}")
        return []
    t0 = time.perf_counter()
    dets = detector.detect(img)
    t_det = time.perf_counter() - t0
    img_np = np.array(img)
    segmenter.set_image(img_np)
    
    t1 = time.perf_counter()
    if dets:
        # Пакетное предсказание масок
        boxes = [d["bbox_xyxy"] for d in dets]
        masks = segmenter.masks_from_boxes_batch(boxes)
        out = []
        H, W = img_np.shape[0], img_np.shape[1]
        for i, (d, mask) in enumerate(zip(dets, masks)):
            # Гарантируем 2D bool маску для последующей обработки
            m = np.asarray(mask)
            if m.dtype != bool:
                m = m != 0
            if m.ndim == 3:
                if m.shape[0] <= 4 and m.shape[0] != m.shape[1]:
                    m = m.any(axis=0)
                else:
                    m = m.any(axis=2)
            if m.ndim != 2 or m.shape != (H, W):
                # Если форма неожиданная — используем пустую маску, чтобы не падать
                m = np.zeros((H, W), dtype=bool)
            tight = compute_tight_bbox_from_mask(m)
            out.append({
                "bbox_xyxy": tight,
                "mask": m,
                "score_vlm": float(d.get("score_vlm", 0.0)),
            })
    else:
        out = []
    t_sam = time.perf_counter() - t1
    
    # Вкладываем тайминги внутрь результатов как метаданные специального ключа
    # (не пойдёт в COCO, используется только для печати после первого прогона)
    if out:
        for d in out:
            d["_timing_det_s"] = t_det
            d["_timing_sam_s"] = t_sam
    return out


def save_results_for_image(img_path: str, class_name: str, dets: List[Dict], out_dir: str, stem: str) -> None:
    """
    Сохраняет результаты обработки одного изображения.
    
    Сохраняет: визуализацию с масками, отдельные маски PNG, JSON с аннотациями.
    Выполняется в фоновом потоке для оптимизации.
    
    Args:
        img_path: Путь к исходному изображению
        class_name: Имя класса инструмента
        dets: Список детекций с масками
        out_dir: Папка для сохранения результатов
        stem: Имя файла без расширения
    """
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"[WARN] Не открыть для сохранения визуализации: {img_path}, {e}")
        img = None

    # Визуализация
    if img is not None:
        overlay_path = os.path.join(out_dir, f"{stem}_ann_with_masks.jpg")
        try:
            save_overlay_with_masks(img, dets, overlay_path, class_name)
        except Exception as e:
            print(f"[WARN] Не удалось сохранить визуализацию: {overlay_path}, {e}")

    # Сохранение масок по каждому объекту
    for idx, d in enumerate(dets):
        try:
            mask = np.asarray(d["mask"], dtype=bool)
            mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
            mask_path = os.path.join(out_dir, f"{stem}_mask_{idx:02d}.png")
            mask_img.save(mask_path)
        except Exception as e:
            print(f"[WARN] Не удалось сохранить маску для {stem} #{idx}: {e}")

    # Пер-изображение JSON (список детекций; mask -> RLE)
    per_img_json = os.path.join(out_dir, f"{stem}.json")
    json_rec = {
        "file_name": os.path.basename(img_path),
        "class": class_name,
        "width": img.width if img else 0,
        "height": img.height if img else 0,
        "detections": [],
    }
    for d in dets:
        try:
            rle = rle_encode_mask_bool(np.asarray(d["mask"], dtype=bool))
        except Exception:
            rle = {"size": [int(json_rec["height"]), int(json_rec["width"])], "counts": []}
        x1, y1, x2, y2 = d["bbox_xyxy"]
        json_rec["detections"].append({
            "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
            "segmentation": rle,
            "score_vlm": float(d.get("score_vlm", 0.0)),
        })
    try:
        with open(per_img_json, "w", encoding="utf-8") as f:
            json.dump(json_rec, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] Не удалось записать JSON: {per_img_json}, {e}")


def process_class_folder(args_tuple):
    """
    Обрабатывает одну папку-класс в отдельном процессе.
    
    Инициализирует модели GroundingDINO и SAM-2, обрабатывает все изображения в классе.
    Использует фоновый поток для сохранения результатов.
    
    Args:
        args_tuple: Кортеж (class_folder, box_thr, text_thr, overwrite[, gpu_id])
    """
    # Распаковка аргументов с обратной совместимостью (4 или 5 значений)
    if len(args_tuple) == 5:
        class_folder, box_thr, text_thr, overwrite, gpu_id = args_tuple
    else:
        class_folder, box_thr, text_thr, overwrite = args_tuple
        gpu_id = None
    
    class_name = os.path.basename(class_folder)
    pid = os.getpid()
    print(f"[PROC {pid}] Запуск обработки папки: {class_name}")
    
    # Фиксация GPU для процесса до любой инициализации CUDA
    if gpu_id is not None:
        # Ограничиваем видимые GPU, чтобы внутри процесса использовать cuda:0
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[PROC {pid}] Назначен GPU {gpu_id} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")

    # Устройства
    device_det, device_seg = find_device_pair()
    
    # Инициализация моделей в процессе
    print(f"[PROC {pid}] Инициализация GroundingDINO: {GROUNDING_DINO_MODEL}")
    detector = GroundingDINODetector(
        model_name=GROUNDING_DINO_MODEL,
        prompt=GROUNDING_PROMPT,
        box_thr=box_thr,
        text_thr=text_thr,
        device=device_det,
    )
    print(f"[PROC {pid}] GroundingDINO инициализирован на {device_det}")
    log_gpu_memory(f"proc_{pid}_after_gdino_init")

    print(f"[PROC {pid}] Инициализация SAM-2 (large)")
    segmenter = SAM2Segmenter(SAM2_CFG, SAM2_CKPT, device_seg)
    print(f"[PROC {pid}] SAM-2 инициализирован на {device_seg}")
    log_gpu_memory(f"proc_{pid}_after_sam2_init")

    out_dir = os.path.join(class_folder, SUBDIR_OUT)
    os.makedirs(out_dir, exist_ok=True)

    # Список изображений
    all_images = list_images(class_folder)
    to_process = []
    for p in all_images:
        stem = os.path.splitext(os.path.basename(p))[0]
        per_img_json = os.path.join(out_dir, f"{stem}.json")
        if not overwrite and os.path.isfile(per_img_json):
            continue
        to_process.append(p)

    print(f"[PROC {pid}] {class_name}: всего изображений={len(all_images)}, к обработке={len(to_process)}, overwrite={overwrite}")

    printed_first_timing = False
    if torch_cuda.is_available():
        torch_cuda.reset_peak_memory_stats()

    # Очередь и фоновый сохранитель
    save_q: "queue.Queue[Optional[Tuple[str, str, List[Dict], str, str]]]" = queue.Queue(maxsize=64)

    def saver_loop() -> None:
        while True:
            item = save_q.get()
            if item is None:
                save_q.task_done()
                break
            img_path_i, class_name_i, dets_i, out_dir_i, stem_i = item
            try:
                save_results_for_image(img_path_i, class_name_i, dets_i, out_dir_i, stem_i)
            except Exception as e:
                print(f"[PROC {pid}] [WARN] Ошибка при сохранении {stem_i}: {e}")
            finally:
                save_q.task_done()

    saver_thread = threading.Thread(target=saver_loop, daemon=True)
    saver_thread.start()

    pbar = tqdm(to_process, desc=f"[{pid}] {class_name}", unit="img", position=pid % 4)
    for img_path in pbar:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        pbar.set_postfix_str(stem)

        t_all0 = time.perf_counter()
        dets = process_image_with_models(img_path, detector, segmenter)

        # Отправляем в фон сохранение
        t_save0 = time.perf_counter()
        try:
            save_q.put((img_path, class_name, dets, out_dir, stem), block=True)
        except Exception as e:
            print(f"[PROC {pid}] [WARN] Не удалось поставить в очередь сохранение {stem}: {e}")
        t_save_enqueue = time.perf_counter() - t_save0

        # Печать таймингов после первого прогона
        if not printed_first_timing:
            t_all = time.perf_counter() - t_all0
            det_time = dets[0].get("_timing_det_s", 0.0) if dets else 0.0
            sam_time = dets[0].get("_timing_sam_s", 0.0) if dets else 0.0
            print(
                f"[PROC {pid}] [TIMING 1st image] detection={det_time:.3f}s, sam={sam_time:.3f}s, enqueue_save={t_save_enqueue:.3f}s, total={t_all:.3f}s"
            )
            log_gpu_memory(f"proc_{pid}_after_first_image")
            printed_first_timing = True

    pbar.close()

    # Корректное завершение сохранителя
    save_q.put(None)
    save_q.join()
    print(f"[PROC {pid}] Завершена обработка папки: {class_name}")


def aggregate_coco_from_jsons(dataset_dir: str, class_folders: List[str]) -> Dict:
    """
    Агрегирует COCO JSON из сохранённых per-image JSON файлов.
    
    Собирает все сохранённые JSON файлы с аннотациями и создаёт
    единый COCO-совместимый файл с категориями, изображениями и аннотациями.
    
    Args:
        dataset_dir: Корневая папка датасета
        class_folders: Список путей к папкам-классам
        
    Returns:
        Словарь COCO с полями 'images', 'annotations', 'categories'
    """
    print("[STAGE] Агрегация COCO JSON из сохранённых файлов...")
    
    categories_map: Dict[str, int] = {}
    categories_list: List[Dict] = []
    images_list: List[Dict] = []
    annotations_list: List[Dict] = []
    ann_id = 1

    for class_folder in tqdm(class_folders, desc="Агрегация COCO", unit="folder"):
        class_name = os.path.basename(class_folder)
        out_dir = os.path.join(class_folder, SUBDIR_OUT)
        
        # Категория -> id
        if class_name not in categories_map:
            categories_map[class_name] = len(categories_map) + 1
            categories_list.append({"id": categories_map[class_name], "name": class_name})

        # Сканируем JSON файлы
        json_files = glob.glob(os.path.join(out_dir, "*.json"))
        for json_path in json_files:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                file_name = data.get("file_name", "")
                width = int(data.get("width", 0))
                height = int(data.get("height", 0))
                detections = data.get("detections", [])
                
                if not file_name:
                    continue
                
                # Попытка определить размеры изображения, если не указаны
                if width == 0 or height == 0:
                    img_path = os.path.join(class_folder, file_name)
                    try:
                        with Image.open(img_path) as img:
                            width, height = img.size
                    except Exception:
                        continue
                
                image_id = len(images_list) + 1
                images_list.append({
                    "id": image_id,
                    "file_name": os.path.relpath(os.path.join(class_folder, file_name), start=dataset_dir),
                    "width": int(width),
                    "height": int(height),
                })

                for det in detections:
                    x1, y1, x2, y2 = det.get("bbox_xyxy", [0, 0, 0, 0])
                    w = max(0.0, float(x2) - float(x1))
                    h = max(0.0, float(y2) - float(y1))
                    rle = det.get("segmentation", {})
                    area = rle_area(rle) if rle else (w * h)
                    
                    annotations_list.append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": categories_map[class_name],
                        "bbox": [float(x1), float(y1), float(w), float(h)],
                        "area": float(area),
                        "iscrowd": 0,
                        "segmentation": rle,
                    })
                    ann_id += 1
                    
            except Exception as e:
                print(f"[WARN] Ошибка чтения {json_path}: {e}")
                continue

    return {
        "images": images_list,
        "annotations": annotations_list,
        "categories": categories_list,
    }


def worker_entry_mgpu(gpu_id: int, folders_subset: List[str], box_thr: float, text_thr: float, overwrite: bool) -> None:
    """
    Точка входа для worker-процесса в multi-GPU режиме.
    
    Обрабатывает подмножество папок-классов на одном GPU.
    Должна быть на верхнем уровне для сериализации multiprocessing.
    
    Args:
        gpu_id: ID GPU для данного worker'а
        folders_subset: Подмножество папок для обработки
        box_thr: Порог для bbox
        text_thr: Порог для текста
        overwrite: Перезаписывать существующие результаты
    """
    # Ограничиваем видимость устройств ДО любой инициализации CUDA/torch
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    pid = os.getpid()
    print(f"[WORKER {pid}] старт на GPU {gpu_id}, всего папок: {len(folders_subset)}")
    for cf in folders_subset:
        # Внутри процесса теперь 'cuda:0' соответствует физическому gpu_id
        process_class_folder((cf, box_thr, text_thr, overwrite))


def worker_entry_single_folder_mgpu(gpu_id: int, class_folder: str, images_subset: List[str], box_thr: float, text_thr: float, overwrite: bool, progress_counter=None) -> None:
    """
    Обработка одного класса: шардинг по изображениям между GPU.
    
    Специальный режим для случая, когда остался один класс с многими изображениями.
    Распараллеливает обработку между GPU по изображениям.
    
    Args:
        gpu_id: ID GPU для данного worker'а
        class_folder: Путь к папке класса
        images_subset: Подмножество изображений для обработки
        box_thr: Порог для bbox
        text_thr: Порог для текста
        overwrite: Перезаписывать существующие результаты
        progress_counter: Общий счётчик прогресса multiprocessing.Value
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    pid = os.getpid()
    class_name = os.path.basename(class_folder)
    print(f"[WORKER {pid}] класс='{class_name}', GPU {gpu_id}, изображений в шардe: {len(images_subset)}")

    # Устройства и модели
    device_det, device_seg = find_device_pair()
    detector = GroundingDINODetector(
        model_name=GROUNDING_DINO_MODEL,
        prompt=GROUNDING_PROMPT,
        box_thr=box_thr,
        text_thr=text_thr,
        device=device_det,
    )
    segmenter = SAM2Segmenter(SAM2_CFG, SAM2_CKPT, device_seg)

    out_dir = os.path.join(class_folder, SUBDIR_OUT)
    os.makedirs(out_dir, exist_ok=True)

    printed_first_timing = False
    if torch_cuda.is_available():
        torch_cuda.reset_peak_memory_stats()

    for img_path in images_subset:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        t_all0 = time.perf_counter()
        dets = process_image_with_models(img_path, detector, segmenter)

        # Синхронное сохранение результата для данного изображения
        t_save0 = time.perf_counter()
        try:
            save_results_for_image(img_path, class_name, dets, out_dir, stem)
        except Exception as e:
            print(f"[WORKER {pid}] [WARN] Ошибка при сохранении {stem}: {e}")
        t_save = time.perf_counter() - t_save0

        if not printed_first_timing:
            det_time = dets[0].get("_timing_det_s", 0.0) if dets else 0.0
            sam_time = dets[0].get("_timing_sam_s", 0.0) if dets else 0.0
            t_all = time.perf_counter() - t_all0
            print(f"[WORKER {pid}] [TIMING 1st image] detection={det_time:.3f}s, sam={sam_time:.3f}s, saving={t_save:.3f}s, total={t_all:.3f}s")
            log_gpu_memory(f"worker_{pid}_after_first_image")
            printed_first_timing = True

        # Обновление общего прогресса
        if progress_counter is not None:
            try:
                with progress_counter.get_lock():
                    progress_counter.value += 1
            except Exception:
                pass


def main():
    """
    Главная функция для аннотации датасета инструментов.
    
    Основные этапы:
    1. Парсинг аргументов командной строки
    2. Сбор статистики по папкам и изображениям
    3. Обработка изображений (последовательно или параллельно)
    4. Агрегация результатов в COCO JSON
    
    Примеры запуска:
    # Базовый запуск (последовательная обработка)
    python annotate_coco_gdino_sam2.py --dataset-dir /path/to/dataset
    
    # Параллельная обработка на 4 процессах
    python annotate_coco_gdino_sam2.py --dataset-dir /path/to/dataset --procs 4
    
    # Multi-GPU обработка на всех доступных GPU
    python annotate_coco_gdino_sam2.py --dataset-dir /path/to/dataset --multi-gpu --procs 8
    
    # Multi-GPU на конкретных GPU с переписыванием существующих результатов
    python annotate_coco_gdino_sam2.py --dataset-dir /path/to/dataset --multi-gpu --gpu-ids "0,1,3" --overwrite
    
    # Только статистика без обработки
    python annotate_coco_gdino_sam2.py --dataset-dir /path/to/dataset --stats-only
    """
    import argparse

    parser = argparse.ArgumentParser(description="Аннотация COCO c GroundingDINO + SAM-2 по подпапкам датасета")
    parser.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET_DIR, help="Путь к папке с классами")
    parser.add_argument("--output-coco", type=str, default=None, help="Путь к COCO JSON (по умолчанию рядом с dataset-dir)")
    parser.add_argument("--overwrite", action="store_true", help="Перезаписывать существующие результаты (по умолчанию нет)")
    parser.add_argument("--box-thr", type=float, default=BOX_THR)
    parser.add_argument("--text-thr", type=float, default=TEXT_THR)
    parser.add_argument("--procs", type=int, default=1, help="Количество процессов (последовательно — 1; мульти-GPU — до числа GPU)")
    parser.add_argument("--multi-gpu", action="store_true", help="Включить многогпу, каждый процесс на своей GPU")
    parser.add_argument("--gpu-ids", type=str, default=None, help="Список GPU через запятую, напр. '0,1'")
    parser.add_argument("--stats-only", action="store_true", help="Только вывести статистику и завершить, без инференса")
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    coco_out_path = args.output_coco or os.path.join(dataset_dir, "coco_annotations.json")

    # Сбор подпапок-категорий
    class_folders = collect_class_folders(dataset_dir)
    if not class_folders:
        print("[FAIL] Не найдено ни одной папки класса в:", dataset_dir)
        sys.exit(1)

    print(f"[STAGE] Найдено папок-категорий: {len(class_folders)}")
    for cf in class_folders:
        print("  -", os.path.basename(cf))

    # Предстартовая статистика по папкам
    print("[STAGE] Предстартовая статистика по папкам:")
    total_imgs_all = 0
    total_done_all = 0
    for cf in class_folders:
        class_name = os.path.basename(cf)
        out_dir = os.path.join(cf, SUBDIR_OUT)
        os.makedirs(out_dir, exist_ok=True)
        all_imgs = list_images(cf)
        total_imgs = len(all_imgs)
        done = 0
        for p in all_imgs:
            stem = os.path.splitext(os.path.basename(p))[0]
            per_img_json = os.path.join(out_dir, f"{stem}.json")
            if os.path.isfile(per_img_json):
                done += 1
        todo = total_imgs - done
        total_imgs_all += total_imgs
        total_done_all += done
        print(f"  - {class_name}: всего={total_imgs}, сделано={done}, осталось={todo}")
    print(f"[TOTAL] всего={total_imgs_all}, сделано={total_done_all}, осталось={total_imgs_all - total_done_all}")

    if args.stats_only:
        print("[MODE] Статистика выведена. Завершение по --stats-only.")
        return

    # Обработка папок
    if not args.multi_gpu:
        if args.procs == 1:
            print("[MODE] Последовательная обработка (1 процесс)")
            for class_folder in class_folders:
                process_class_folder((class_folder, args.box_thr, args.text_thr, args.overwrite))
        else:
            print(f"[MODE] Параллельная обработка (Pool, {args.procs} процессов)")
            mp.set_start_method('spawn', force=True)
            task_args = [(cf, args.box_thr, args.text_thr, args.overwrite) for cf in class_folders]
            with mp.Pool(processes=args.procs) as pool:
                pool.map(process_class_folder, task_args)
                pool.close()
                pool.join()
    else:
        # Мульти-GPU: по одному процессу на каждую указанную GPU
        print("[MODE] Multi-GPU: один процесс на GPU, последовательность папок внутри процесса")
        mp.set_start_method('spawn', force=True)

        # Разбор списка GPU
        available_gpu_count = torch.cuda.device_count()
        if available_gpu_count == 0:
            print("[FAIL] Запрошен multi-gpu, но CUDA не доступна")
            sys.exit(1)

        if args.gpu_ids is not None:
            gpu_ids_list = [int(x.strip()) for x in args.gpu_ids.split(',') if x.strip() != '']
        else:
            gpu_ids_list = list(range(available_gpu_count))

        if not gpu_ids_list:
            print("[FAIL] Пустой список GPU")
            sys.exit(1)

        # Кол-во воркеров равно мин(procs, число указанных GPU)
        num_workers = max(1, min(int(args.procs), len(gpu_ids_list)))
        gpu_ids_list = gpu_ids_list[:num_workers]
        print(f"[MGPU] Используем GPU: {gpu_ids_list} (workers={num_workers})")

        # Проверяем, осталась ли фактически одна папка с незавершёнными изображениями
        remaining_folders: List[str] = []
        remaining_counts: Dict[str, int] = {}
        for cf in class_folders:
            out_dir_cf = os.path.join(cf, SUBDIR_OUT)
            imgs_cf = list_images(cf)
            todo_cf = 0
            for pth in imgs_cf:
                st = os.path.splitext(os.path.basename(pth))[0]
                if not os.path.isfile(os.path.join(out_dir_cf, f"{st}.json")):
                    todo_cf += 1
            if todo_cf > 0:
                remaining_folders.append(cf)
                remaining_counts[cf] = todo_cf

        if len(remaining_folders) == 1:
            # Спец-режим: шардим одну папку по изображениям между GPU
            target_folder = remaining_folders[0]
            print(f"[MGPU] Обнаружена одна папка для обработки: '{os.path.basename(target_folder)}', шардим по изображениям")
            all_imgs = list_images(target_folder)
            out_dir_tf = os.path.join(target_folder, SUBDIR_OUT)
            to_process_imgs: List[str] = []
            for pth in all_imgs:
                st = os.path.splitext(os.path.basename(pth))[0]
                if args.overwrite or not os.path.isfile(os.path.join(out_dir_tf, f"{st}.json")):
                    to_process_imgs.append(pth)

            if not to_process_imgs:
                print("[MGPU] В выбранной папке нечего обрабатывать. Выходим.")
            else:
                # Разбиваем список изображений на num_workers чанков
                chunks: List[List[str]] = [[] for _ in range(num_workers)]
                for i, pth in enumerate(to_process_imgs):
                    chunks[i % num_workers].append(pth)

                # Единый прогресс-бар в главном процессе
                total_images = len(to_process_imgs)
                progress_counter = mp.Value('i', 0)
                pbar = tqdm(total=total_images, desc=f"[MGPU] {os.path.basename(target_folder)}", unit="img", position=0)

                procs: List[mp.Process] = []
                for i in range(num_workers):
                    if not chunks[i]:
                        continue
                    p = mp.Process(target=worker_entry_single_folder_mgpu, args=(gpu_ids_list[i], target_folder, chunks[i], args.box_thr, args.text_thr, args.overwrite, progress_counter))
                    p.start()
                    procs.append(p)

                # Обновляем pbar по счетчику
                last_val = 0
                while any(p.is_alive() for p in procs):
                    try:
                        with progress_counter.get_lock():
                            current_val = progress_counter.value
                    except Exception:
                        current_val = last_val
                    if current_val > last_val:
                        pbar.update(current_val - last_val)
                        last_val = current_val
                    time.sleep(0.1)

                # Добираем остаток и закрываем pbar
                try:
                    with progress_counter.get_lock():
                        current_val = progress_counter.value
                except Exception:
                    current_val = last_val
                if current_val > last_val:
                    pbar.update(current_val - last_val)
                pbar.close()

                for p in procs:
                    p.join()
        else:
            # Обычный режим: распределяем папки по воркерам round-robin
            buckets: List[List[str]] = [[] for _ in range(num_workers)]
            for idx, cf in enumerate(class_folders):
                buckets[idx % num_workers].append(cf)

            procs: List[mp.Process] = []
            for i in range(num_workers):
                p = mp.Process(target=worker_entry_mgpu, args=(gpu_ids_list[i], buckets[i], args.box_thr, args.text_thr, args.overwrite))
                p.start()
                procs.append(p)
            for p in procs:
                p.join()

    # Агрегация COCO JSON из сохранённых файлов
    coco_obj = aggregate_coco_from_jsons(dataset_dir, class_folders)
    
    with open(coco_out_path, "w", encoding="utf-8") as f:
        json.dump(coco_obj, f, ensure_ascii=False, indent=2)
    print("✅ COCO JSON сохранён:", coco_out_path)


if __name__ == "__main__":
    main()


