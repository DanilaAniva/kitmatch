import torch
from PIL import Image, ImageDraw
from transformers import BatchEncoding, GroundingDinoProcessor
from transformers.models.mm_grounding_dino import MMGroundingDinoForObjectDetection
import os

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

name = "iSEE-Laboratory/llmdet_base"
processor = GroundingDinoProcessor.from_pretrained(name)
model = MMGroundingDinoForObjectDetection.from_pretrained(
    name,
    low_cpu_mem_usage=True
).to(device)

def detect_objects(
    image_path: str,
    model: MMGroundingDinoForObjectDetection,
    processor: GroundingDinoProcessor
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """
    Обнаруживает объекты на изображении с помощью модели MMGroundingDinoForObjectDetection.

    Args:
        image_path: Путь к изображению.
        model: Модель MMGroundingDinoForObjectDetection.
        processor: Процессор для обработки изображений.

    Returns:
        box: Координаты бокса.
        score: Уверенность.
        label: Класс.
    """
    image = Image.open(image_path)
    texts = [["tool"]]
    inputs: BatchEncoding = processor(text=texts, images=image, return_tensors="pt")
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Размеры исходного изображения (высота, ширина) для масштабирования предсказаний
    target_sizes = torch.Tensor([image.size[::-1]])
    # Преобразование предсказаний (координаты боксов и классы) в формат Pascal VOC (xmin, ymin, xmax, ymax)
    results: list[dict] = processor.post_process_grounded_object_detection(
    outputs=outputs, 
    target_sizes=target_sizes, 
    threshold=0.1
    )

    boxes: torch.Tensor = results[0]["boxes"]
    scores: torch.Tensor = results[0]["scores"]
    labels: list[str] = results[0]["text_labels"]

    # Получаем индекс с максимальной уверенностью
    max_idx: int = int(torch.argmax(scores))
    box: torch.Tensor = boxes[max_idx]
    score: torch.Tensor = scores[max_idx]
    label: str = labels[max_idx]

    return box, score, label


def draw_boxes_on_image(image_path: str, box: torch.Tensor) -> Image.Image:
    """
    Рисуем прямоугольники по координатам на изображении.

    Args:
        image_path: Путь к изображению, на котором рисуем.
        box: Тензор координат в формате (xmin, ymin, xmax, ymax).

    Returns:
        Копия изображения с нарисованным боксом.
    """
    vis_image = Image.open(image_path)
    draw = ImageDraw.Draw(vis_image)

    xmin, ymin, xmax, ymax = box.cpu().tolist()
    draw.rectangle([(xmin-5, ymin-5), (xmax+5, ymax+5)], outline=(255, 0, 0), width=3)

    return vis_image


def save_vertical_crop(image_path: str, box: torch.Tensor, output_path: str) -> str:
    """
    Вырезаем объект по боксу, поворачиваем в вертикальную ориентацию и сохраняем.

    Что делаем: Вырезаем область по координатам бокса и поворачиваем на 90°, если ширина больше высоты.
    Зачем делаем: Чтобы получить отдельное изображение объекта, ориентированное вертикально.

    Args:
        image_path: Путь к исходному изображению.
        box: Координаты бокса (xmin, ymin, xmax, ymax) в тензоре.
        output_path: Путь для сохранения результата.

    Returns:
        Путь к сохраненному изображению.
    """
    image = Image.open(image_path)
    xmin, ymin, xmax, ymax = [int(v) for v in box.cpu().tolist()]
    crop = image.crop((xmin, ymin, xmax, ymax))
    width, height = crop.size
    if width > height:
        crop = crop.rotate(90, expand=True)
    crop.save(output_path)
    return output_path
from typing import Tuple
from PIL import Image
import numpy as np
import cv2 as cv


def mask_with_grabcut(tool_img: Image.Image, border_ratio: float = 0.08, iters: int = 5) -> Tuple[Image.Image, Image.Image]:
    """
    Построить маску инструмента с помощью GrabCut, предполагая, что картинка уже обрезана боксом.

    Parameters
    ----------
    tool_img : Image.Image
        Обрезанное изображение инструмента (желательно, чтобы по краям присутствовал фон).
    border_ratio : float
        Доля ширины/высоты, считающаяся заведомым фоном по периметру (рамка).
        Например, 0.08 означает 8% от меньшей стороны — уйдёт в "точно фон".
    iters : int
        Количество итераций алгоритма GrabCut.

    Returns
    -------
    mask_L : Image.Image
        Бинарная маска инструмента в режиме 'L' (0=фон, 255=инструмент).
    cutout_rgba : Image.Image
        Исходное изображение с альфа-каналом по маске (режим 'RGBA').

    Notes
    -----
    - Работает без нейросети, устойчив при аккуратном кропе, где фон заметен по краям.
    - Если инструмент упирается в края (почти нет фона), алгоритму сложнее.
    """
    rgb = tool_img.convert("RGB")
    img = np.array(rgb, dtype=np.uint8)
    h, w = img.shape[:2]

    # Инициализируем маску: по периметру "точно фон", центр — "вероятно объект"
    mask = np.full((h, w), cv.GC_PR_FGD, dtype=np.uint8)

    border = max(5, int(border_ratio * min(h, w)))
    mask[:border, :] = cv.GC_BGD
    mask[-border:, :] = cv.GC_BGD
    mask[:, :border] = cv.GC_BGD
    mask[:, -border:] = cv.GC_BGD

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Запускаем GrabCut в режиме initWithMask
    cv.grabCut(img, mask, None, bgdModel, fgdModel, iters, mode=cv.GC_INIT_WITH_MASK)

    # Собираем бинарную маску: объект = {FGD, PR_FGD}
    mask_bin = np.where((mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD), 255, 0).astype(np.uint8)

    # Небольшая морфология для сглаживания (по желанию)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    mask_bin = cv.morphologyEx(mask_bin, cv.MORPH_OPEN, kernel, iterations=1)
    mask_bin = cv.morphologyEx(mask_bin, cv.MORPH_CLOSE, kernel, iterations=1)

    mask_L = Image.fromarray(mask_bin, mode="L")
    cutout_rgba = rgb.copy()
    cutout_rgba.putalpha(mask_L)
    return mask_L, cutout_rgba
import io
from PIL import Image
from rembg import remove, new_session
from rembg.sessions.base import BaseSession

def cutout_to_mask(img: Image.Image | bytes | bytearray, session: BaseSession) -> Tuple[Image.Image, Image.Image]:
    """
    Преобразует вырезанное изображение в бинарную маску.

    Args:
        img: Входное изображение.
        session: Сессия для удаления фона.
    """
    rgb = img.convert("RGB")
    out_data = remove(rgb, session=session)
    # Приводим к RGBA
    if isinstance(out_data, (bytes, bytearray)):
        out_rgba = Image.open(io.BytesIO(out_data)).convert("RGBA")
    else:  # PIL.Image.Image
        out_rgba = out_data.convert("RGBA")
    # Берём альфа-канал как маску
    alpha = out_rgba.getchannel("A")
    # Опционально: жёсткая бинаризация 0/255
    mask_L = alpha.point(lambda a: 255 if a > 0 else 0, mode="L")
    return mask_L, out_rgba
import os
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import traceback
import random
import time
import argparse

# Конфигурация путей (значения по умолчанию, могут быть переопределены CLI)
DEFAULT_DATASET_CANDIDATES = [
    Path("Dataset"),
    Path("dataset"),
]
# Поддиректории для результатов внутри каждой классовой папки
SUBDIR_CROPPED = "cropped"
SUBDIR_MASKS = "masks"
SUBDIR_CUTOUTS = "cutouts"

# Какие расширения исходных изображений обрабатывать
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

# Глобальные параметры сегментации
GRABCUT_BORDER_RATIO = 0.05
GRABCUT_ITERS = 7
SESSION = new_session("u2net")

# Ограничение количества изображений на класс и зерно случайности (может быть переопределено CLI)
MAX_PER_CLASS = 20
RANDOM_SEED = 42


def ensure_dir(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def resolve_dataset_dir(cli_path: str | None) -> Path:
    if cli_path:
        p = Path(cli_path)
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {p.resolve()}")
        return p
    for cand in DEFAULT_DATASET_CANDIDATES:
        if cand.exists() and cand.is_dir():
            return cand
    raise FileNotFoundError(
        "Dataset directory not found. Tried: " + ", ".join(str(c) for c in DEFAULT_DATASET_CANDIDATES)
    )


def process_class_folder(class_dir: Path, max_per_class: int) -> None:
    """
    Для папки класса создаёт три подпапки и обрабатывает (до max_per_class случайных) изображений:
    - Сохраняет вертикальный кроп: <basename>_cropped<ext> в подпапке 'cropped'
    - Сохраняет маску инструмента: <basename>_mask.png в подпапке 'masks'
    - Сохраняет вырезанный инструмент с альфой: <basename>_cutout.png в подпапке 'cutouts'
    """
    cropped_dir = class_dir / SUBDIR_CROPPED
    masks_dir = class_dir / SUBDIR_MASKS
    cutouts_dir = class_dir / SUBDIR_CUTOUTS

    for d in (cropped_dir, masks_dir, cutouts_dir):
        ensure_dir(d)

    # Список исходных изображений на верхнем уровне класса
    image_files: List[Path] = [
        p for p in sorted(class_dir.iterdir())
        if p.is_file() and p.suffix in IMAGE_EXTENSIONS
    ]

    # Выборка случайных max_per_class файлов (если их больше)
    if len(image_files) > max_per_class and max_per_class != float('inf'):
        random.seed(RANDOM_SEED)
        selected_files = random.sample(image_files, max_per_class)
        print(f"Sampling {len(selected_files)} of {len(image_files)} images in {class_dir.name}")
    else:
        selected_files = image_files
        print(f"Using all {len(selected_files)} images in {class_dir.name}")

    for entry in selected_files:
        ext = entry.suffix
        basename = entry.stem
        try:
            time_start = time.time()
            # 1) Обнаружение бокса и вертикальный кроп
            box, score, label = detect_objects(str(entry), model, processor)
            cropped_path = cropped_dir / f"{basename}_cropped{ext}"
            save_vertical_crop(str(entry), box, str(cropped_path))
            image = Image.open(str(entry))
            xmin, ymin, xmax, ymax = [int(v) for v in box.cpu().tolist()]
            crop = image.crop((xmin, ymin, xmax, ymax))
            crop.save(str(cropped_path))
            time_end = time.time()
            print(f"Time for detect_objects: {time_end - time_start} seconds")

            time_start = time.time()
            # 2) Маска и вырезка через GrabCut ИЛИ rembg (RemoveBackGround)
            cropped_img = Image.open(str(cropped_path))
            # mask_img, cutout_img = mask_with_grabcut(cropped_img, border_ratio=GRABCUT_BORDER_RATIO, iters=GRABCUT_ITERS) # GrabCut
            mask_img, cutout_img = cutout_to_mask(cropped_img, SESSION) # rembg
            time_end = time.time()
            print(f"Time for mask_with_grabcut: {time_end - time_start} seconds")

            mask_path = masks_dir / f"{basename}_mask.png"  # маску надёжнее хранить как PNG
            cutout_path = cutouts_dir / f"{basename}_cutout.png"  # RGBA также удобно в PNG

            mask_img.save(str(mask_path))
            cutout_img.save(str(cutout_path))

            print(f"OK: {entry.name} -> {cropped_path.name}, {mask_path.name}, {cutout_path.name}")
        except Exception as e:
            print(f"ERROR processing {entry}: {e}")
            traceback.print_exc()


def main() -> None:
    global MAX_PER_CLASS
    parser = argparse.ArgumentParser(description="Process a dataset of tools into cropped/masks/cutouts")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Путь к папке с классами (по умолчанию ищем Dataset/ или dataset/)" )
    parser.add_argument("--max-per-class", type=int, default=MAX_PER_CLASS, help="Максимум изображений на класс")
    parser.add_argument("--no-limit", action="store_true", help="Обработать все изображения без ограничений")
    parser.add_argument("--exclude-folders", nargs="*", default=[], help="Список папок для исключения (по имени)")
    args = parser.parse_args()

    if args.no_limit:
        MAX_PER_CLASS = float('inf')  # Без ограничений
    else:
        MAX_PER_CLASS = args.max_per_class
    
    dataset_dir = resolve_dataset_dir(args.dataset_dir)
    exclude_set = set(args.exclude_folders)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir.resolve()}")

    all_folders = [p for p in sorted(dataset_dir.iterdir()) if p.is_dir()]
    class_folders = [p for p in all_folders if p.name not in exclude_set]
    
    print(f"Device: {device}")
    print(f"Using model: {name}")
    print(f"Dataset dir: {dataset_dir.resolve()}")
    print(f"Found {len(all_folders)} total folders, {len(class_folders)} after exclusions")
    if exclude_set:
        print(f"Excluded folders: {', '.join(exclude_set)}")
    print(f"Max per class: {'Unlimited' if MAX_PER_CLASS == float('inf') else MAX_PER_CLASS}")

    for class_dir in class_folders:
        print(f"Processing class folder: {class_dir}")
        process_class_folder(class_dir, MAX_PER_CLASS)

    print("Done.")


if __name__ == "__main__":
    main()
