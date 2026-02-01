"""
Тест реалистичной аугментации с перекрытием инструментов

Этот скрипт генерирует синтетические примеры для тестирования качества детекции 
инструментов в условиях частичного перекрытия другими объектами. Используется для 
валидации алгоритмов детекции и сегментации в реалистичных сценариях.

Основная логика:
1. Загружает RGBA изображения инструментов из галереи
2. Генерирует композитные сцены: базовый инструмент + случайные заслоняющие объекты
3. Применяет реалистичные аугментации: размытие, JPEG артефакты
4. Тестирует качество детекции через GroundingDINO + SAM-2
5. Сравнивает предсказанные маски с ground truth через IoU метрику
6. Сохраняет визуализации и метрики для анализа

Модели:
- GroundingDINO (rziga/mm_grounding_dino_large_all) для детекции
- SAM-2 для генерации масок
"""

import os, random, argparse, json, sys, time
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image, ImageFilter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


def setup_logging(out_dir: str):
    """
    Настраивает логгинг в консоль и файл.
    
    Args:
        out_dir: Папка для сохранения лог-файла
        
    Returns:
        Объект logger
    """
    import logging
    os.makedirs(out_dir, exist_ok=True)
    log = logging.getLogger("occlusion_test")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(out_dir, "occlusion_test.log"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch.setFormatter(fmt); fh.setFormatter(fmt)
    log.addHandler(ch); log.addHandler(fh)
    return log


def list_rgba_images(folder: str) -> List[str]:
    """
    Находит все изображения в папке.
    
    Args:
        folder: Путь к папке
        
    Returns:
        Отсортированный список путей к файлам
    """
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    files = []
    for f in sorted(os.listdir(folder)):
        p = os.path.join(folder, f)
        if os.path.isfile(p) and f.lower().endswith(exts):
            files.append(p)
    return files


def load_rgba(path: str) -> Image.Image:
    """
    Загружает изображение и преобразует в RGBA формат.
    
    Args:
        path: Путь к файлу изображения
        
    Returns:
        PIL изображение в RGBA формате
    """
    im = Image.open(path)
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    return im


def resize_to_fit(im: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """
    Изменяет размер изображения с сохранением пропорций.
    
    Вписывает изображение в целевой размер без искажения.
    
    Args:
        im: Исходное изображение
        target_size: Целевой размер (ширина, высота)
        
    Returns:
        Изменённое изображение
    """
    tw, th = target_size
    w, h = im.size
    scale = min(tw / max(1, w), th / max(1, h))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return im.resize((nw, nh), Image.BICUBIC)


def random_position_overlap(base_bbox: Tuple[int, int, int, int], occ_size: Tuple[int, int], canvas_size: Tuple[int, int], min_overlap: float = 0.2) -> Tuple[int, int]:
    """
    Находит случайную позицию для заслоняющего объекта с минимальным перекрытием.
    
    Пытается найти позицию, где заслоняющий объект перекрывает
    базовый объект на заданный процент.
    
    Args:
        base_bbox: Координаты базового объекта (x1, y1, x2, y2)
        occ_size: Размер заслоняющего объекта (ширина, высота)
        canvas_size: Размер канвы (ширина, высота)
        min_overlap: Минимальная доля перекрытия (0.0-1.0)
        
    Returns:
        Координаты позиции (x, y) для размещения заслоняющего объекта
    """
    bw, bh = base_bbox[2] - base_bbox[0], base_bbox[3] - base_bbox[1]
    ow, oh = occ_size
    cw, ch = canvas_size
    # Желаемая зона вокруг базы
    x_min = max(0, base_bbox[0] - ow // 2)
    y_min = max(0, base_bbox[1] - oh // 2)
    x_max = min(cw - ow, base_bbox[2])
    y_max = min(ch - oh, base_bbox[3])
    if x_max < x_min: x_min, x_max = 0, max(0, cw - ow)
    if y_max < y_min: y_min, y_max = 0, max(0, ch - oh)
    # Несколько попыток найти позицию с пересечением
    for _ in range(20):
        x = random.randint(x_min, max(x_min, x_max)) if (x_max >= x_min) else 0
        y = random.randint(y_min, max(y_min, y_max)) if (y_max >= y_min) else 0
        inter_w = max(0, min(x + ow, base_bbox[2]) - max(x, base_bbox[0]))
        inter_h = max(0, min(y + oh, base_bbox[3]) - max(y, base_bbox[1]))
        if inter_w * inter_h >= min_overlap * max(1, bw * bh):
            return x, y
    # Фолбек — произвольная позиция
    return random.randint(0, max(0, cw - ow)), random.randint(0, max(0, ch - oh))


def composite_sample(base: Image.Image, base_name: str, occluders_named: List[Tuple[str, Image.Image]], canvas_size: int = 512,
                     blur_bg: bool = False, jpeg_artifacts: bool = False) -> Tuple[Image.Image, Dict]:
    """
    Создаёт композитную сцену с базовым инструментом и заслоняющими объектами.
    
    Размещает базовый объект по центру канвы, затем добавляет
    1-3 случайных заслоняющих объекта с перекрытием.
    
    Args:
        base: Основной RGBA инструмент
        base_name: Имя файла базового инструмента
        occluders_named: Список кортежей (имя, RGBA_изображение)
        canvas_size: Размер квадратной канвы
        blur_bg: Применять размытие фона
        jpeg_artifacts: Применять JPEG артефакты
        
    Returns:
        Кортеж (композитное_изображение_RGB, метаданные_объектов)
    """
    # Канвас и фон
    canvas = Image.new("RGB", (canvas_size, canvas_size), (127, 127, 127))
    # Подготовим базовый объект по центру
    base_scaled = resize_to_fit(base, (int(canvas_size * 0.9), int(canvas_size * 0.9)))
    bw, bh = base_scaled.size
    bx = (canvas_size - bw) // 2
    by = (canvas_size - bh) // 2
    canvas_rgba = canvas.convert("RGBA")
    canvas_rgba.alpha_composite(base_scaled, (bx, by))
    base_bbox = (bx, by, bx + bw, by + bh)
    base_alpha = np.array(base_scaled)[..., 3]
    base_mask_canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    base_mask_canvas[by:by+bh, bx:bx+bw] = (base_alpha > 127).astype(np.uint8) * 255
    objects = []
    objects.append({
        "class": os.path.splitext(os.path.basename(base_name))[0],
        "role": "base",
        "bbox": [int(base_bbox[0]), int(base_bbox[1]), int(base_bbox[2]), int(base_bbox[3])],
        "mask": base_mask_canvas,
    })

    # Случайные заслоняющие объекты
    num_occ = random.randint(1, max(1, min(3, len(occluders_named))))
    chosen = random.sample(occluders_named, num_occ)
    for occ_name, occ in chosen:
        scale = random.uniform(0.35, 0.85)
        target_w = int(canvas_size * scale)
        target_h = int(canvas_size * scale)
        occ_scaled = resize_to_fit(occ, (target_w, target_h))
        ow, oh = occ_scaled.size
        ox, oy = random_position_overlap(base_bbox, (ow, oh), (canvas_size, canvas_size), min_overlap=random.uniform(0.2, 0.6))
        canvas_rgba.alpha_composite(occ_scaled, (ox, oy))
        occ_alpha = np.array(occ_scaled)[..., 3]
        m = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        m[oy:oy+oh, ox:ox+ow] = (occ_alpha > 127).astype(np.uint8) * 255
        objects.append({
            "class": os.path.splitext(os.path.basename(occ_name))[0],
            "role": "occluder",
            "bbox": [int(ox), int(oy), int(ox + ow), int(oy + oh)],
            "mask": m,
        })

    out = canvas_rgba.convert("RGB")
    if blur_bg and random.random() < 0.3:
        out = out.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    if jpeg_artifacts and random.random() < 0.3:
        # Пережимаем в JPEG на лету
        import io
        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=random.randint(60, 92))
        buf.seek(0)
        out = Image.open(buf).convert("RGB")
    meta = {"canvas_size": int(canvas_size), "objects": objects}
    return out, meta


# --------- GroundingDINO + SAM-2 ----------

def init_gdino(model_name: str, log=None):
    """
    Инициализирует модель GroundingDINO.
    
    Args:
        model_name: Название модели на HuggingFace Hub
        log: Опциональный logger для вывода
        
    Returns:
        Кортеж (processor, detector_model)
    """
    if log:
        log.info(f"[INIT] Инициализация GroundingDINO: {model_name}")
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    start_time = time.time()
    proc = AutoProcessor.from_pretrained(model_name)
    det = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).eval()
    elapsed = time.time() - start_time
    if log:
        log.info(f"[OK] GroundingDINO загружен за {elapsed:.2f}s")
    return proc, det


def run_gdino_boxes(proc, det, image_pil: Image.Image, prompt: str = "tool", box_thr: float = 0.25, text_thr: float = 0.25) -> List[Dict]:
    """
    Выполняет детекцию объектов через GroundingDINO.
    
    Args:
        proc: Процессор от transformers
        det: Модель детектора
        image_pil: Изображение PIL
        prompt: Текстовый промпт
        box_thr: Порог для bbox
        text_thr: Порог для текста
        
    Returns:
        Список словарей с bbox и скорами
    """
    inputs = proc(images=[image_pil], text=[prompt + " ."], return_tensors="pt")
    with torch.no_grad():
        outputs = det(**inputs)
    results = proc.post_process_grounded_object_detection(outputs=outputs, input_ids=inputs.input_ids,
                                                          threshold=box_thr, text_threshold=text_thr,
                                                          target_sizes=[image_pil.size[::-1]])
    res = results[0]
    out = []
    boxes = res.get("boxes", [])
    scores = res.get("scores", [])
    for i in range(min(len(boxes), len(scores))):
        bb = boxes[i].detach().float().cpu().tolist()
        sc = float(scores[i].detach().cpu())
        out.append({"bbox": [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])], "score": sc})
    return out


def init_sam2(cfg: str, ckpt: str, log=None):
    """
    Инициализирует модель SAM-2.
    
    Args:
        cfg: Путь к YAML конфигу SAM-2
        ckpt: Путь к checkpoint файлу SAM-2
        log: Опциональный logger
        
    Returns:
        Объект SAM2ImagePredictor
    """
    if log:
        log.info(f"[INIT] Инициализация SAM-2: {cfg} + {ckpt}")
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    start_time = time.time()
    model = build_sam2(cfg, ckpt)
    pred = SAM2ImagePredictor(model)
    elapsed = time.time() - start_time
    if log:
        log.info(f"[OK] SAM-2 загружен за {elapsed:.2f}s")
    return pred


def mask_from_box_sam2(predictor, img_np: np.ndarray, xyxy: List[float]) -> np.ndarray:
    """
    Генерирует маску объекта по bounding box через SAM-2.
    
    Args:
        predictor: Инициализированный SAM2ImagePredictor
        img_np: Изображение в формате numpy [H, W, 3]
        xyxy: Координаты bounding box [x1, y1, x2, y2]
        
    Returns:
        Булева маска [H, W]
    """
    predictor.set_image(img_np)
    b = np.array(xyxy, dtype=np.float32)
    masks, ious, _ = predictor.predict(box=b[None, :], multimask_output=True)
    if masks.shape[0] == 0:
        return np.zeros((img_np.shape[0], img_np.shape[1]), dtype=bool)
    best = int(np.argmax(ious.reshape(-1)))
    return masks[best].astype(bool)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """
    Вычисляет Intersection over Union (IoU) между двумя масками.
    
    Args:
        a: Первая маска
        b: Вторая маска
        
    Returns:
        Значение IoU от 0.0 до 1.0
    """
    a = a.astype(bool); b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def save_overlay(image_rgb: Image.Image, objects: List[Dict], out_path: str) -> None:
    """
    Сохраняет визуализацию с наложенными масками объектов.
    
    Каждый класс отображается своим цветом на полупрозрачном оверлее.
    
    Args:
        image_rgb: Исходное RGB изображение
        objects: Список объектов с масками и классами
        out_path: Путь для сохранения
    """
    H, W = image_rgb.height, image_rgb.width
    overlay = np.zeros((H, W, 4), dtype=np.uint8)
    # Цвет по классу
    unique = list(dict.fromkeys([o["class"] for o in objects]))
    cmap = plt.cm.Set3(np.linspace(0, 1, max(len(unique), 1)))
    class_to_color = {cls: (np.array(cmap[i % len(cmap)][:3]) * 255).astype(np.uint8) for i, cls in enumerate(unique)}
    for obj in objects:
        m = obj.get("mask_pred") if obj.get("role") == "base" else obj.get("mask")
        if m is None:
            m = obj.get("mask")
        if m is None:
            continue
        rgb = class_to_color.get(obj["class"], np.array([255, 0, 0], dtype=np.uint8))
        overlay[m.astype(bool)] = [rgb[0], rgb[1], rgb[2], 120]
    img = np.array(image_rgb)
    out = Image.fromarray(img).convert("RGBA")
    out.alpha_composite(Image.fromarray(overlay, mode="RGBA"))
    out.convert("RGB").save(out_path)


def main():
    """
    Главная функция для тестирования аугментации с перекрытием инструментов.
    
    Выполняет полный цикл: генерация синтетических сцен → детекция → сегментация → оценка качества.
    
    Примеры запуска:
    # Базовый запуск с 20 примерами
    python occlusion_augmentation_test.py
    
    # Генерация 50 примеров с кастомными папками
    python occlusion_augmentation_test.py --input_dir /path/to/rgba/tools --out_dir /path/to/output --num 50
    
    # Использование другой модели GroundingDINO
    python occlusion_augmentation_test.py --grounding_dino "IDEA-Research/grounding-dino-base" --box_thr 0.3
    
    # Размер канвы 768x768 с другими порогами
    python occlusion_augmentation_test.py --canvas 768 --box_thr 0.2 --text_thr 0.2
    """
    ap = argparse.ArgumentParser(description="Тест реалистичной аугментации перекрытия инструментами")
    ap.add_argument("--input_dir", type=str, default="/home/ubuntu/diabert/dataset/crops_of_every_tool/Кропнутые инструменты все",
                    help="Папка с RGBA вырезками инструментов")
    ap.add_argument("--out_dir", type=str, default="/home/ubuntu/diabert/dataset/predrazmetka_dashi/occlusion_tests",
                    help="Куда сохранять примеры")
    ap.add_argument("--num", type=int, default=20, help="Сколько примеров сгенерировать")
    ap.add_argument("--canvas", type=int, default=512, help="Размер канвы (квадрат)")
    # GDINO + SAM2
    ap.add_argument("--grounding_dino", type=str, default="rziga/mm_grounding_dino_large_all")
    ap.add_argument("--box_thr", type=float, default=0.25)
    ap.add_argument("--text_thr", type=float, default=0.25)
    ap.add_argument("--sam2_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--sam2_ckpt", type=str, default="/home/ubuntu/sam2/checkpoints/sam2.1_hiera_large.pt")
    args = ap.parse_args()

    log = setup_logging(args.out_dir)
    log.info("🚀 Запуск теста реалистичных перекрытий инструментов")
    
    log.info(f"[STAGE] Поиск RGBA изображений в {args.input_dir}")
    paths = list_rgba_images(args.input_dir)
    if len(paths) < 2:
        log.error(f"Недостаточно изображений в {args.input_dir}")
        raise SystemExit(1)
    log.info(f"[OK] Найдено {len(paths)} RGBA изображений")

    log.info("[STAGE] Загрузка изображений...")
    images = []
    for p in tqdm(paths, desc="Загрузка RGBA", unit="img"):
        try:
            images.append((p, load_rgba(p)))
        except Exception as e:
            log.warning(f"Пропуск {p}: {e}")
    log.info(f"[OK] Загружено {len(images)} изображений")

    # Инициализация моделей
    log.info("[STAGE] Инициализация моделей...")
    try:
        proc, det = init_gdino(args.grounding_dino, log)
        sam2 = init_sam2(args.sam2_cfg, args.sam2_ckpt, log)
    except Exception as e:
        log.error(f"Не удалось инициализировать GroundingDINO/SAM-2: {e}")
        raise SystemExit(1)

    log.info(f"[STAGE] Генерация {args.num} примеров...")
    pbar = tqdm(range(args.num), desc="Генерация", unit="img")
    for i in pbar:
        pbar.set_postfix(stage="композит")
        base_name, base_img = random.choice(images)
        base_class = os.path.splitext(os.path.basename(base_name))[0]
        # Формируем список окклюдеров (имя, изображение)
        occluders_named = images
        
        start_time = time.time()
        img, meta = composite_sample(base_img, base_name, occluders_named, canvas_size=args.canvas, blur_bg=True, jpeg_artifacts=True)
        comp_time = time.time() - start_time
        
        stem = f"example_{i:02d}"
        out_img = os.path.join(args.out_dir, stem + ".jpg")
        img.save(out_img)

        pbar.set_postfix(stage="детекция")
        # Предсказание маски базового объекта через GD+SAM-2
        start_time = time.time()
        dets = run_gdino_boxes(proc, det, img, prompt="tool", box_thr=args.box_thr, text_thr=args.text_thr)
        det_time = time.time() - start_time
        
        base_gt_mask = None
        for o in meta["objects"]:
            if o["role"] == "base":
                base_gt_mask = (np.asarray(o["mask"]) > 127)
                break
        
        pbar.set_postfix(stage="сегментация")
        base_pred_mask = None
        sam_time = 0.0
        if dets and base_gt_mask is not None:
            img_np = np.array(img)
            best_iou = -1.0
            start_time = time.time()
            for d in dets:
                m = mask_from_box_sam2(sam2, img_np, d["bbox"])  # bool HxW
                val = iou(m, base_gt_mask)
                if val > best_iou:
                    best_iou = val
                    base_pred_mask = m
            sam_time = time.time() - start_time
            log.info(f"  [{i:02d}] {base_class}: {len(dets)} dets, лучший IoU={best_iou:.3f}, время: композит={comp_time:.2f}s детект={det_time:.2f}s SAM={sam_time:.2f}s")
        
        pbar.set_postfix(stage="сохранение")
        # Сохраним маски: для базового — предсказанную и GT, для остальных — GT
        masks_dir = os.path.join(args.out_dir, stem + "_masks")
        os.makedirs(masks_dir, exist_ok=True)
        # Базовый
        if base_gt_mask is not None:
            Image.fromarray((base_gt_mask.astype(np.uint8) * 255)).save(os.path.join(masks_dir, "base_gt.png"))
        if base_pred_mask is not None:
            Image.fromarray((base_pred_mask.astype(np.uint8) * 255)).save(os.path.join(masks_dir, "base_pred_gdsam.png"))
        # Остальные
        for idx, o in enumerate(meta["objects"]):
            if o["role"] == "occluder":
                m = np.asarray(o["mask"]) > 127
                clsname = o["class"].replace("/", "_")
                Image.fromarray((m.astype(np.uint8) * 255)).save(os.path.join(masks_dir, f"occ_{idx:02d}_{clsname}.png"))

        # Визуализация наложения масок
        if base_pred_mask is not None:
            for o in meta["objects"]:
                if o["role"] == "base":
                    o["mask_pred"] = base_pred_mask
                    break
        vis_path = os.path.join(args.out_dir, stem + "_vis.jpg")
        save_overlay(img, meta["objects"], vis_path)

        # JSON с описанием объектов
        json_out = {
            "image": os.path.basename(out_img),
            "objects": [
                {
                    "class": o["class"],
                    "role": o["role"],
                    "bbox": o["bbox"],
                } for o in meta["objects"]
            ]
        }
        with open(os.path.join(args.out_dir, stem + ".json"), "w", encoding="utf-8") as f:
            json.dump(json_out, f, ensure_ascii=False, indent=2)
    
    pbar.close()
    log.info(f"✅ Готово! Результаты в {args.out_dir}")


if __name__ == "__main__":
    main()


