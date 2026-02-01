"""
Автоматическая аннотация инструментов с GroundingDINO, SAM-2 и обученным DINOv3 эмбеддером

Этот скрипт выполняет комплексную автоматическую аннотацию изображений инструментов,
объединяя три нейронные сети для максимальной точности детекции и классификации:

1. GroundingDINO для детекции bounding boxes по текстовому промпту "tool"
2. SAM-2 для генерации точных масок сегментации по найденным boxes
3. Обученный DINOv3 эмбеддер для классификации объектов через сравнение с галереей эталонов

Основная логика:
1. Загружает галерею PNG эталонов и вычисляет их эмбеддинги
2. Для каждого входного изображения:
   - Запускает GroundingDINO детекцию с NMS фильтрацией
   - Генерирует маски через SAM-2 для каждого найденного bbox
   - Вырезает и маскирует области объектов
   - Сравнивает эмбеддинги с галереей через косинусное сходство
   - Применяет алгоритм уникальности классов (один класс = один объект на изображении)
3. Сохраняет результаты в JSON (native или COCO формат)
4. Создаёт детальные визуализации с bbox, масками и цветовым кодированием
5. Экспортирует отладочные артефакты: кропы, маски, карты похожести

Поддерживает:
- Обученные чекпойнты EmbedNet или vanilla DINOv3 из timm
- Мульти-GPU распределение и модель-параллелизм
- Асинхронное сохранение результатов
- Детальное логирование времени выполнения этапов

Модели:
- GroundingDINO (rziga/mm_grounding_dino_large_all) для детекции
- SAM-2 (hiera_large) для сегментации
- DINOv3 (vit_large_patch16_dinov3.lvd1689m) или обученный EmbedNet для классификации
"""

import os, sys, glob, json, time, csv, shutil
import unicodedata
from datetime import datetime
from typing import List, Tuple, Dict
import gc
import psutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn.functional as F
import timm
from tqdm import tqdm
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from threading import Thread
import uuid # Добавлено для генерации уникальных ID для предсказаний Label Studio
try:
    from joblib import Parallel, delayed
    print(f"[INFO] joblib используется для Label Studio")
except Exception as e:
    print(f"[WARN] joblib не используется для Label Studio: {e}")
    Parallel = None
    delayed = None
try:
    import orjson
except Exception:
    orjson = None

# ====== КОНСТАНТЫ ДЛЯ ЗАПОЛНЕНИЯ ======
IMAGES_DIR   = "/home/ubuntu/diabert/dataset/dataset/Групповые для тренировки"  # папка с изображениями для разметки
NUM_IMAGES   = 600
GALLERY_DIR  = "/home/ubuntu/diabert/dataset/crops_of_every_tool/Кропнутые инструменты все"  # PNG эталоны
BASE_DIR     = "/home/ubuntu/diabert/dataset/predrazmetka_dashi/dinov3-sam2-gdino_10_try"  # папка для результатов эксперимента

# Модель эмбеддингов (самая большая DINOv3 из timm)
# TIMM_DINOV3_ID = "vit_7b_patch16_dinov3.lvd1689m"
TIMM_DINOV3_ID = "vit_large_patch16_dinov3.lvd1689m"

# GroundingDINO (детектор боксов) и текстовый промпт
GROUNDING_DINO_MODEL = "rziga/mm_grounding_dino_large_all"
GROUNDING_PROMPT     = "tool"
BOX_THR = 0.25
TEXT_THR = 0.25

# NMS для детекций GroundingDINO
APPLY_NMS_BEFORE_SAM = True
NMS_IOU_THR = 0.5
# Предпочтение при конфликте: "score" (выше score_vlm) или "area" (больший бокс)
NMS_PREFERENCE = "score"

# SAM-2 конфиг и веса (самый большой)
SAM2_CFG  = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT = "/home/ubuntu/sam2/checkpoints/sam2.1_hiera_large.pt"

SAVE_VIS = False  # флаг: сохранять ли визуализации (_ann.jpg, _ann_recalculated.jpg, _ann_with_masks.jpg)
JSON_SAVE_TQDM = True  # флаг: показывать прогресс tqdm при сохранении JSON
JSON_SAVE_ASYNC = False  # флаг: сохранять JSON асинхронно в отдельном потоке
OUTPUT_FORMAT = "label_studio"  # варианты: "native" | "coco" | "label_studio"
JSON_CHECKPOINT_EVERY = 20  # каждые N изображений сохранять промежуточный predictions.json
JSON_SPLIT_SIZE = 100  # разбивать JSON на части по N записей для избежания проблем с большими файлами

# ====== ОТЛАДКА И ФЛАГИ DINOv3 ======
# Детальный экспорт промежуточных артефактов
DEBUG_EXPORT = False
# Использовать модель-параллелизм DINOv3 на 2 GPU
DINO_USE_MODEL_PARALLEL = False
# В одногпу режиме пробовать FP16 для экономии памяти
DINO_SINGLE_GPU_FP16 = False
# Форсировать использование одной GPU (cuda:0) для всех моделей
FORCE_SINGLE_GPU = True

# Индекс видеокарты для режима FORCE_SINGLE_GPU (например, 1 если 0 занята)
SINGLE_GPU_INDEX = 1

# Минимальный косинус для назначения нового класса при перераспределении
# Значения < 0 считаем слишком плохими и такие классы не присваиваем
UNIQUE_REASSIGN_MIN_SCORE = 0.0
# Переназначать ли дубли на другие свободные классы.
# Если False — дубли будут просто удалены (сохраняем только лучший экземпляр).
UNIQUE_REASSIGN_DUPLICATES = False

# =========================================================================
# ДИНАМИЧЕСКИЙ/ПЕРЕЗАПУСКАЕМЫЙ BASE_DIR
# Если задана переменная окружения RESUME_EXP_DIR — продолжаем в этой папке,
# иначе создаём новую директорию с текущей временной меткой.
RESUME_EXP_DIR = os.getenv("RESUME_EXP_DIR", "").strip()
if RESUME_EXP_DIR:
    BASE_DIR = RESUME_EXP_DIR
else:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    BASE_DIR = f"/home/ubuntu/diabert/dataset/predrazmetka_dashi/dinov3-sam2-gdino_{timestamp}"
# =========================================================================


def _canon_class(name: str) -> str:
    """
    Канонизация имени класса для унификации строк.
    
    Применяет Unicode нормализацию NFC и обрезает пробелы.
    Используется для сопоставления классов из разных источников.
    
    Параметры:
        name (str): исходное имя класса
        
    Возвращает:
        str: нормализованное имя класса
    """
    try:
        return unicodedata.normalize("NFC", str(name)).strip()
    except Exception:
        return str(name).strip()

# ====== ДОБАВЛЕНО: опции использования обученного чекпойнта EmbedNet ======
# Если указан путь к чекпойнту и конфигу, используем EmbedNet вместо timm DINOv3
DINO3_USE_CHECKPOINT = True
DINO3_CKPT = "/home/ubuntu/diabert/dataset/predrazmetka_dashi/lct_dino3_out/run_009/checkpoint_last.pth"  # /path/to/run_xxx/checkpoint_best.pth
DINO3_EMB_DIM = 128
DINO3_MODEL_ID = "vit_large_patch16_dinov3.lvd1689m"
DINO3_IMG_SIZE = 224


# Распределение моделей по устройствам
num_gpus = torch.cuda.device_count()
if FORCE_SINGLE_GPU and torch.cuda.is_available():
    # Все на одной выбранной GPU
    idx = int(SINGLE_GPU_INDEX)
    if idx < 0 or idx >= num_gpus:
        print(f"[WARN] Некорректный индекс GPU {idx}; доступно {num_gpus}. Использую 0.")
        idx = 0
    print(f"[INFO] Форсируем один GPU для всех моделей: cuda:{idx}")
    dev = torch.device(f"cuda:{idx}")
    device_embed = dev
    device_det = dev
    device_seg = dev
    dino_mp_second_device = None
else:
    if num_gpus > 1:
        print(f"[INFO] Найдено {num_gpus} GPU. Модели будут распределены.")
        device_embed = torch.device("cuda:0")
        device_det = torch.device("cuda:1")
        device_seg = torch.device("cuda:1")
        dino_mp_second_device = torch.device("cuda:1")
    else:
        single_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Найдено {num_gpus} GPU. Используется одно устройство: {single_device}")
        device_embed = single_device
        device_det = single_device
        device_seg = single_device
        dino_mp_second_device = None


# Папка эксперимента (все артефакты в одном месте)
EXP_DIR = BASE_DIR
os.makedirs(EXP_DIR, exist_ok=True)

# ====== ИМПОРТЫ БИБЛИОТЕК ======
try:
    from transformers import AutoImageProcessor, AutoModel
except Exception:
    print("[WARN] transformers не используется для DINOv3, но может быть нужен для других частей")

try:
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
except Exception:
    raise SystemExit("Не найден GroundingDINO в transformers. Установите пакет transformers >=4.40 и проверьте модель rziga/mm_grounding_dino_large_all")

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
        raise SystemExit("SAM-2 не найден. Установите его из /home/ubuntu/sam2: pip install -e '.[notebooks]'")


# ====== КЛАССЫ ======
def measure_time(func):
    """
    Декоратор для измерения времени выполнения функции.
    
    Оборачивает функцию для замера времени выполнения с высокой точностью.
    Используется для профилирования этапов обработки изображений.
    
    Параметры:
        func: функция для декорирования
        
    Возвращает:
        function: обёрнутая функция, которая возвращает (result, elapsed_seconds)
    """
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        res = func(*args, **kwargs)
        dt = time.perf_counter() - t0
        return res, dt
    return wrapper

# ====== DINOv3 EMBEDDER (HuggingFace) ======
class DinoV3Embedder:
    """
    Эмбеддер на базе DINOv3 из библиотеки timm.
    
    Предназначен для извлечения L2-нормированных эмбеддингов признаков из изображений.
    Использует предобученную модель DINOv3 для получения дискриминативных признаков
    объектов, которые затем используются для классификации через косинусное сходство.
    
    Параметры:
        model_id (str): идентификатор модели в timm (например, "vit_large_patch16_dinov3.lvd1689m")
        device (torch.device): устройство для выполнения (CPU/CUDA)
        second_device (torch.device | None): не используется в этой реализации
        use_fp16 (bool): использовать ли FP16 для экономии памяти на GPU
        
    Методы:
        embed(img: PIL.Image) -> torch.Tensor: извлекает эмбеддинг размерности [D]
    """

    def __init__(self, model_id: str, device: torch.device, second_device: torch.device | None = None, use_fp16: bool = True):
        self.device = device
        self.second_device = None  # явно отключаем внутренний MP, используем отдельный класс ниже
        self.model = timm.create_model(model_id, pretrained=True, num_classes=0).to(self.device).eval()
        if self.device.type == "cuda" and use_fp16:
            self.model = self.model.half()
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)
        print(f"[INFO] DinoV3 (timm, single-device) модель загружена на {self.device}")
        print("[INFO] DinoV3 (timm) transform:", self.transforms)

    @torch.no_grad()
    def embed(self, img: Image.Image) -> torch.Tensor:
        """
        Извлекает эмбеддинг из изображения.
        
        Применяет трансформы, прогоняет через DINOv3 модель и нормализует результат.
        
        Параметры:
            img (PIL.Image): входное изображение (автоматически конвертируется в RGB)
            
        Возвращает:
            torch.Tensor: L2-нормированный эмбеддинг размерности [D] на CPU в float32
        """
        if img.mode != "RGB":
            img = img.convert("RGB")
        t_img = self.transforms(img).unsqueeze(0).to(self.device)
        # Автокаст включаем только если запросили fp16
        use_autocast = (self.device.type == "cuda" and any(p.dtype == torch.float16 for p in self.model.parameters()))
        if use_autocast:
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                feats = self.model(t_img)
        else:
            feats = self.model(t_img)
        feats = F.normalize(feats, dim=1)
        # Возвращаем в float32 на CPU, чтобы избегать проблем с fp16 на CPU и NaN
        return feats.squeeze(0).detach().cpu().float()


class DinoV3Embedder2GPU:
    """
    DINOv3 эмбеддер с пайплайн-параллелизмом на двух GPU.
    
    Разделяет трансформерные блоки DINOv3 между двумя GPU для ускорения
    больших моделей. Первая половина блоков выполняется на dev0,
    вторая половина на dev1 с передачей активаций между устройствами.
    
    Параметры:
        model_id (str): идентификатор модели в timm
        dev0 (torch.device): первое CUDA устройство для начальных блоков
        dev1 (torch.device): второе CUDA устройство для финальных блоков  
        split_at (int | None): номер блока для разделения (по умолчанию половина)
        dtype: тип данных для весов модели (по умолчанию float32)
        
    Методы:
        embed(img: PIL.Image) -> torch.Tensor: извлекает эмбеддинг с пайплайн-параллелизмом
    """

    def __init__(self, model_id: str, dev0: torch.device, dev1: torch.device, split_at: int | None = None, dtype=torch.float32):
        assert dev0.type == "cuda" and dev1.type == "cuda", "Нужны две CUDA-карты"
        self.dev0 = dev0
        self.dev1 = dev1
        self.dtype = dtype

        # 1) Создаём модель на CPU, чтобы не словить OOM при .to()
        self.model = timm.create_model(model_id, pretrained=True, num_classes=0)  # остаётся на CPU
        self.model.eval()

        # Определяем количество блоков и точку сплита
        assert hasattr(self.model, "blocks"), "Ожидался ViT с атрибутом `blocks`"
        L = len(self.model.blocks)
        self.split = split_at if split_at is not None else L // 2
        self.split = max(1, min(self.split, L - 1))

        # 2) Переносим части модели на нужные девайсы
        # Передняя часть на dev0
        self.model.patch_embed.to(self.dev0, dtype=self.dtype)
        if hasattr(self.model, "pos_drop") and self.model.pos_drop is not None:
            self.model.pos_drop.to(self.dev0)
        if hasattr(self.model, "cls_token") and self.model.cls_token is not None:
            self.model.cls_token = torch.nn.Parameter(self.model.cls_token.detach().to(self.dev0, dtype=self.dtype))
            print(f"[MP] cls_token -> {self.dev0}, dtype={self.dtype}")
        if hasattr(self.model, "pos_embed") and self.model.pos_embed is not None:
            self.model.pos_embed = torch.nn.Parameter(self.model.pos_embed.detach().to(self.dev0, dtype=self.dtype))
            print(f"[MP] pos_embed -> {self.dev0}, dtype={self.dtype}")
        for i in range(self.split):
            self.model.blocks[i].to(self.dev0, dtype=self.dtype)
            print(f"[MP] block[{i}] -> {self.dev0}, dtype={self.dtype}")

        # Задняя часть на dev1
        for i in range(self.split, L):
            self.model.blocks[i].to(self.dev1, dtype=self.dtype)
            print(f"[MP] block[{i}] -> {self.dev1}, dtype={self.dtype}")
        if hasattr(self.model, "norm") and self.model.norm is not None:
            self.model.norm.to(self.dev1, dtype=self.dtype)
            print(f"[MP] norm -> {self.dev1}, dtype={self.dtype}")
        if hasattr(self.model, "fc_norm") and self.model.fc_norm is not None:
            self.model.fc_norm.to(self.dev1, dtype=self.dtype)
            print(f"[MP] fc_norm -> {self.dev1}, dtype={self.dtype}")
        if hasattr(self.model, "pre_logits") and self.model.pre_logits is not None:
            self.model.pre_logits.to(self.dev1, dtype=self.dtype)
            print(f"[MP] pre_logits -> {self.dev1}, dtype={self.dtype}")

        # 3) Трансформы — обычные timm transforms (на CPU)
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)
        print(f"[INFO] DinoV3 шардирован: blocks[0..{self.split-1}] на {self.dev0}, blocks[{self.split}..{L-1}] на {self.dev1}")

    @torch.inference_mode()
    def _forward_sharded(self, x: torch.Tensor) -> torch.Tensor:
        """
        Выполняет forward pass с шардингом между двумя GPU.
        
        Повторяет логику forward_features для ViT, но с переносом активаций
        между устройствами на границе блоков. Первая часть на dev0,
        вторая часть на dev1.
        
        Параметры:
            x (torch.Tensor): входной тензор изображения на dev0 с типом self.dtype
            
        Возвращает:
            torch.Tensor: финальные признаки на dev1
        """
        m = self.model
        B = x.shape[0]

        print(f"[MP] input on {self.dev0}, shape={x.shape}, dtype={x.dtype}")

        # patch_embed + добавление cls/pos на dev0
        x = m.patch_embed(x)
        # Приводим к форме (B, N, C) токенов, учитывая разные варианты вывода patch_embed
        if x.dim() == 4:
            # Возможные формы: (B, C, H, W) или (B, H, W, C)
            if hasattr(m, "embed_dim") and x.shape[1] == getattr(m, "embed_dim", x.shape[1]):
                # (B, C, H, W) -> (B, H*W, C)
                B, C, H, W = x.shape
                x = x.flatten(2).transpose(1, 2).contiguous()
                print(f"[MP] patch_embed (B,C,H,W) -> tokens (B,{H*W},{C})")
            elif x.shape[-1] == getattr(m, "embed_dim", x.shape[-1]):
                # (B, H, W, C) -> (B, H*W, C)
                B, H, W, C = x.shape
                x = x.view(B, H * W, C).contiguous()
                print(f"[MP] patch_embed (B,H,W,C) -> tokens (B,{H*W},{C})")
            else:
                raise RuntimeError(f"Unexpected 4D patch_embed output shape: {x.shape}, cannot infer embed_dim")
        elif x.dim() != 3:
            raise RuntimeError(f"Unexpected patch_embed output dims: {x.shape}")

        if hasattr(m, "cls_token") and m.cls_token is not None:
            cls_tokens = m.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        if hasattr(m, "pos_embed") and m.pos_embed is not None:
            # Добавляем только если совпадает длина последовательности
            if m.pos_embed.shape[1] == x.shape[1]:
                x = x + m.pos_embed
            else:
                print(f"[MP] skip pos_embed add: pos_embed_len={m.pos_embed.shape[1]} != seq_len={x.shape[1]}")
        if hasattr(m, "pos_drop") and m.pos_drop is not None:
            x = m.pos_drop(x)

        # Первая половина блоков на dev0
        for i in range(self.split):
            x = m.blocks[i](x)

        # Перенос активации на dev1
        x = x.to(self.dev1, non_blocking=True)
        print(f"[MP] activation moved to {self.dev1}, shape={x.shape}, dtype={x.dtype}")

        # Вторая половина блоков + norm на dev1
        for i in range(self.split, len(m.blocks)):
            x = m.blocks[i](x)
        if hasattr(m, "norm") and m.norm is not None:
            x = m.norm(x)

        # Пулинг/норм согласно конфигу модели
        gp = getattr(m, "global_pool", None)
        if gp == "avg":
            x = x[:, 1:].mean(dim=1)
            if hasattr(m, "fc_norm") and m.fc_norm is not None:
                x = m.fc_norm(x)
        else:
            x = x[:, 0]
            if hasattr(m, "pre_logits") and m.pre_logits is not None:
                x = m.pre_logits(x)
        print(f"[MP] output on {self.dev1}, shape={x.shape}, dtype={x.dtype}")
        return x

    @torch.inference_mode()
    def embed(self, img: Image.Image) -> torch.Tensor:
        """
        Извлекает эмбеддинг с использованием пайплайн-параллелизма.
        
        Применяет трансформы на CPU, затем выполняет forward pass
        с шардингом между двумя GPU.
        
        Параметры:
            img (PIL.Image): входное изображение (конвертируется в RGB)
            
        Возвращает:
            torch.Tensor: L2-нормированный эмбеддинг на CPU в float32
        """
        if img.mode != "RGB":
            img = img.convert("RGB")
        # Готовим вход и отправляем на dev0 в нужном dtype
        t_img = self.transforms(img).unsqueeze(0).to(self.dev0)
        feats = self._forward_sharded(t_img)
        feats = F.normalize(feats, dim=1)
        # Возвращаем в float32 на CPU, чтобы избегать проблем с fp16 на CPU и NaN
        return feats.squeeze(0).detach().cpu().float()


# ====== EmbedNet из тренировочного скрипта ======
class EmbedNet(torch.nn.Module):
    """
    Обученная модель эмбеддингов на базе DINOv3 backbone.
    
    Состоит из замороженного DINOv3 backbone и обучаемой головы для 
    метрического обучения с ArcFace loss. Создаёт эмбеддинги фиксированной
    размерности для классификации инструментов.
    
    Параметры:
        dim (int): размерность выходного эмбеддинга
        model_name (str): название backbone модели в timm
        pool (str): тип global pooling ("token" или "avg") 
        freeze_backbone (bool): заморозить ли веса backbone
        img_size (int | None): размер входного изображения
        
    Методы:
        forward(x: torch.Tensor) -> torch.Tensor: прямой проход через сеть
    """
    
    def __init__(self, dim: int, model_name: str, pool: str = "token", freeze_backbone: bool = True, img_size: int | None = 224):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool=pool,
            img_size=img_size,
        )
        if hasattr(self.backbone, "patch_embed"):
            try:
                self.backbone.patch_embed.strict_img_size = False
            except Exception:
                pass
        feat_dim = self.backbone.num_features
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.head = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через модель эмбеддингов.
        
        Извлекает признаки через backbone, применяет голову и нормализует результат.
        
        Параметры:
            x (torch.Tensor): батч изображений [B, C, H, W]
            
        Возвращает:
            torch.Tensor: L2-нормированные эмбеддинги [B, dim]
        """
        f = self.backbone(x)
        z = torch.nn.functional.normalize(self.head(f), dim=1)
        return z


# ====== ГАЛЕРЕЯ ЭТАЛОНОВ ======
def load_gallery(embedder: DinoV3Embedder, gdir: str) -> Tuple[List[str], torch.Tensor, List[str]]:
    """
    Загружает галерею эталонных изображений и строит матрицу эмбеддингов.
    
    Сканирует папку с эталонами инструментов, загружает каждое изображение,
    извлекает эмбеддинги через указанный эмбеддер и собирает в матрицу
    для последующего поиска по косинусному сходству.
    
    Параметры:
        embedder (DinoV3Embedder): инициализированный эмбеддер для извлечения признаков
        gdir (str): путь к папке с эталонными изображениями (PNG/JPG/WEBP/BMP)
        
    Возвращает:
        Tuple[List[str], torch.Tensor, List[str]]: 
            - список имён эталонов (без расширения)
            - матрица эмбеддингов [N, D]
            - список полных путей к файлам эталонов
    """
    img_paths = sorted([p for p in glob.glob(os.path.join(gdir, "*")) if p.lower().endswith((".png",".jpg",".jpeg",".webp",".bmp"))])
    if not img_paths:
        raise SystemExit(f"В галерее нет изображений: {gdir}")
    names, vecs, paths = [], [], []
    for p in img_paths:
        try:
            with Image.open(p).convert("RGBA") as im:
                z = embedder.embed(im)
            names.append(os.path.splitext(os.path.basename(p))[0])
            vecs.append(z.cpu())
            paths.append(p)
        except Exception as e:
            print("skip gallery", p, e)
    if not vecs:
        raise SystemExit("Не удалось создать эмбеддинги галереи")
    G = torch.stack(vecs, dim=0)
    return names, G, paths


# ====== GroundingDINO детекция ======
class GroundingDINODetector:
    """Обёртка для GroundingDINO из transformers.

    Назначение:
      Получение детекций (bbox) по текстовому промпту.

    Параметры:
      model_name (str): идентификатор модели в HF
      prompt (str): текстовый промпт (например, "tool")
      box_thr (float), text_thr (float): пороги постобработки
      device (torch.device): устройство

    Методы:
      detect(image: PIL.Image) -> List[dict]: список детекций с bbox и score_vlm
    """

    def __init__(self, model_name: str, prompt: str, box_thr: float, text_thr: float, device: torch.device):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(device).eval()
        self.prompt_txt = " . ".join([prompt]) + " ."
        self.box_thr = box_thr
        self.text_thr = text_thr
        self.device = device

    def detect(self, image: Image.Image) -> List[Dict]:
        """Выполняет детекцию на изображении.

        Вход:
          image: PIL.Image RGB
        Выход:
          список словарей: {bbox_xyxy, score_vlm, class}
        """
        inputs = self.processor(images=[image], text=[self.prompt_txt], return_tensors="pt").to(self.device)
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=(self.device.type=="cuda")):
                outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs, input_ids=inputs.input_ids,
            threshold=self.box_thr, text_threshold=self.text_thr,
            target_sizes=[image.size[::-1]],
        )
        res = results[0]
        boxes  = res.get("boxes", [])
        scores = res.get("scores", [])
        dets=[]
        for i in range(min(len(boxes), len(scores))):
            bb = boxes[i].detach().float().cpu().tolist()
            sc = float(scores[i].detach().cpu())
            dets.append({"bbox_xyxy": [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])], "score_vlm": sc, "class": "tool"})

        # Применяем NMS при необходимости
        if APPLY_NMS_BEFORE_SAM and len(dets) > 1:
            dets = self._nms(dets)
        return dets

    @staticmethod
    def _iou_xyxy(a: List[float], b: List[float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
        union = area_a + area_b - inter + 1e-12
        return float(inter / union)

    @staticmethod
    def _box_area_xyxy(b: List[float]) -> float:
        x1, y1, x2, y2 = b
        return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))

    def _nms(self, dets: List[Dict]) -> List[Dict]:
        if not dets:
            return dets
        # Сортировка по предпочтению
        if NMS_PREFERENCE == "area":
            dets_sorted = sorted(dets, key=lambda d: self._box_area_xyxy(d["bbox_xyxy"]), reverse=True)
        else:
            dets_sorted = sorted(dets, key=lambda d: float(d.get("score_vlm", 0.0)), reverse=True)

        kept: List[Dict] = []
        for d in dets_sorted:
            keep = True
            for k in kept:
                iou = self._iou_xyxy(d["bbox_xyxy"], k["bbox_xyxy"])
                if iou >= NMS_IOU_THR:
                    keep = False
                    break
            if keep:
                kept.append(d)
        if len(dets) != len(kept):
            try:
                print(f"[NMS] GroundingDINO: {len(dets)} -> {len(kept)} (removed {len(dets)-len(kept)})")
            except Exception:
                pass
        return kept


# ====== SAM-2: маска по боксу ======
class SAM2Segmenter:
    """Обёртка для SAM‑2 (получение масок по боксу).

    Назначение:
      По входному изображению и bbox получить маску объекта.

    Параметры:
      cfg (str): путь к YAML конфигу SAM-2
      ckpt (str): путь к весам SAM-2
      device (torch.device): устройство

    Методы:
      set_image(img_np: np.ndarray): подготовка изображения
      mask_from_box(xyxy: List[float]) -> np.ndarray[H,W] bool: маска
      tight_bbox_from_mask(mask) -> List[float]: tight bbox по маске
      apply_mask_and_crop(img_pil, mask) -> PIL.Image RGBA: вырез с альфой
    """

    def __init__(self, cfg: str, ckpt: str, device: torch.device):
        self.model = build_sam2(cfg, ckpt).to(device).eval()
        self.predictor = SAM2ImagePredictor(self.model)
        self.device = device
        self._img_np = None

    def set_image(self, img_np: np.ndarray) -> None:
        """Устанавливает изображение для предсказания масок."""
        self._img_np = img_np
        self.predictor.set_image(img_np)

    def mask_from_box(self, xyxy: List[float]) -> np.ndarray:
        """Возвращает булеву маску по заданному bbox (XYXY)."""
        assert self._img_np is not None, "Сначала вызовите set_image()"
        b = np.array(xyxy, dtype=np.float32)
        masks, ious, _ = self.predictor.predict(box=b[None, :], multimask_output=True)
        if masks.shape[0] == 0:
            return np.zeros((self._img_np.shape[0], self._img_np.shape[1]), dtype=bool)
        best = int(np.argmax(ious.reshape(-1)))
        return masks[best].astype(bool)

    @staticmethod
    def tight_bbox_from_mask(mask: np.ndarray) -> List[float]:
        """Считает tight bbox по маске (XYXY)."""
        ys, xs = np.where(mask)
        if xs.size == 0 or ys.size == 0:
            return [0.0, 0.0, 0.0, 0.0]
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        return [float(x1), float(y1), float(x2+1), float(y2+1)]

    @staticmethod
    def apply_mask_and_crop(img_pil: Image.Image, mask: np.ndarray) -> Image.Image:
        """Вырезает объект по маске с сохранением альфа‑канала."""
        ys, xs = np.where(mask)
        if xs.size == 0 or ys.size == 0:
            return img_pil
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        crop = img_pil.crop((x1, y1, x2+1, y2+1)).convert("RGBA")
        alpha = (mask[y1:y2+1, x1:x2+1].astype(np.uint8) * 255)
        rgba = np.array(crop)
        rgba[..., 3] = alpha
        return Image.fromarray(rgba)


def match(z: torch.Tensor, gallery_vecs: torch.Tensor, gallery_names: List[str]) -> Tuple[str, float]:
    """
    Находит ближайший эталон по косинусному сходству.
    
    Выполняет поиск наиболее похожего эталона из галереи для данного
    эмбеддинга объекта. Применяет нормализацию и обработку NaN/Inf значений
    для стабильности вычислений.
    
    Параметры:
        z (torch.Tensor): эмбеддинг детектированного объекта [D]
        gallery_vecs (torch.Tensor): матрица эмбеддингов галереи [N, D]
        gallery_names (List[str]): имена эталонов длины N
        
    Возвращает:
        Tuple[str, float]: имя лучшего эталона и косинусное сходство с ним
    """
    # Приводим к float32 и устраняем NaN/Inf
    z = torch.nan_to_num(z.float(), nan=0.0, posinf=0.0, neginf=0.0)
    g = torch.nan_to_num(gallery_vecs.float(), nan=0.0, posinf=0.0, neginf=0.0).to(z.device)
    # Нормализация с eps во избежание деления на ноль
    if z.dim() != 1:
        z = z.view(-1)
    z = F.normalize(z, dim=0, eps=1e-12)
    g = F.normalize(g, dim=1, eps=1e-12)
    sims = torch.mv(g, z)
    k = int(torch.argmax(sims).item())
    return gallery_names[k], float(sims[k].item())


def assign_unique_classes(det_entries: List[Dict]) -> List[Dict]:
    """
    Перераспределяет классы для обеспечения уникальности на изображении.
    
    Реализует двухэтапный алгоритм для устранения дублирования классов:
    1) Для каждого класса оставляет только объект с максимальным скором
    2) Переназначает оставшиеся объекты на свободные классы из их ranked-списка
    
    Используется для обеспечения того, чтобы каждый класс инструмента был
    представлен не более чем одним объектом на изображении.
    
    Параметры:
        det_entries (List[Dict]): список детекций с полями 'class', 'ranked', 'score_metric'
        
    Возвращает:
        List[Dict]: отфильтрованный список детекций с уникальными классами
    """
    n = len(det_entries)
    if n == 0:
        return []

    # Подсортировать индексы по лучшему скору
    def best_score(idx: int) -> float:
        r = det_entries[idx].get("ranked") or []
        if r:
            return float(r[0].get("score", det_entries[idx].get("score_metric", 0.0)))
        return float(det_entries[idx].get("score_metric", 0.0))

    # Группировка по текущему классу
    class_to_indices: Dict[str, List[int]] = {}
    for i, d in enumerate(det_entries):
        cname = _canon_class(d.get("class", "") or "")
        class_to_indices.setdefault(cname, []).append(i)

    assigned_indices: set[int] = set()
    used_classes: set[str] = set()
    result = [dict(x) for x in det_entries]

    # Этап 1: в каждой группе оставляем лучший по скору
    for cname, idxs in class_to_indices.items():
        if not idxs:
            continue
        # выбрать лучший индекс по скору
        best_idx = max(idxs, key=best_score)
        if cname:
            used_classes.add(cname)
        assigned_indices.add(best_idx)

    # Этап 2: перераспределяем оставшиеся (не назначенные)
    remaining = [i for i in range(n) if i not in assigned_indices]
    remaining.sort(key=best_score, reverse=True)
    kept_indices: set[int] = set(assigned_indices)

    if UNIQUE_REASSIGN_DUPLICATES:
        for i in remaining:
            ranked = det_entries[i].get("ranked") or []
            picked = False
            for cand in ranked:
                cname = _canon_class(cand.get("name"))
                try:
                    cscore = float(cand.get("score", 0.0))
                except Exception:
                    cscore = 0.0
                # Назначаем только свободный класс и только при достаточно хорошем косинусе
                if cname and (cname not in used_classes) and math.isfinite(cscore) and (cscore >= UNIQUE_REASSIGN_MIN_SCORE):
                    result[i]["class"] = cname
                    result[i]["score_metric"] = cscore
                    used_classes.add(cname)
                    kept_indices.add(i)
                    picked = True
                    break
            if not picked:
                # не удалось найти свободный класс — удаляем этот объект из результата
                pass
    else:
        # Не переназначаем дубли — просто удаляем их
        pass

    # Собираем только оставленные объекты
    final = [result[i] for i in sorted(kept_indices)]
    return final


def save_ann_visualization_matplotlib(img: Image.Image, det_entries: List[Dict], out_path: str) -> None:
    """
    Сохраняет визуализацию детекций с bounding boxes.
    
    Создаёт изображение с наложенными bbox, используя уникальные цвета
    для каждого класса и легенду для удобства интерпретации результатов.
    
    Параметры:
        img (PIL.Image): исходное изображение
        det_entries (List[Dict]): список детекций с полями 'bbox_xyxy', 'class', 'score_metric'
        out_path (str): путь для сохранения визуализации
        
    Возвращает:
        None
    """
    class_names = [d.get('class', 'unknown') for d in det_entries]
    unique_classes = list(dict.fromkeys(class_names))
    colors = plt.cm.Set3(np.linspace(0, 1, max(len(unique_classes), 1)))
    class_to_color = {cls: colors[i % len(colors)] for i, cls in enumerate(unique_classes)}

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.imshow(np.array(img))

    for d in det_entries:
        x1,y1,x2,y2 = d["bbox_xyxy"]
        w = max(0.0, float(x2) - float(x1))
        h = max(0.0, float(y2) - float(y1))
        cls_name = d.get('class', 'unknown')
        score = d.get('score_metric', None)
        color = class_to_color.get(cls_name, (1.0, 0.0, 0.0, 1.0))

        rect = patches.Rectangle((x1, y1), w, h, linewidth=3,
                                 edgecolor=color, facecolor='none', alpha=0.9)
        ax.add_patch(rect)

        label = cls_name if (score is None or not (score==score and math.isfinite(score))) else f"{cls_name} {score:.2f}"
        ax.text(x1, max(0, y1 - 10), label, fontsize=8, color='white',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.85),
                fontweight='bold')

    ax.set_xlim(0, img.width)
    ax.set_ylim(img.height, 0)
    ax.axis('off')

    # Легенда по классам (цвет -> класс), чтобы цвета были понятны
    if unique_classes:
        legend_elements = [patches.Patch(color=class_to_color[c], label=c) for c in unique_classes]
        # Размещаем легенду справа сверху, чуть наружу поля изображения
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=10)
    plt.tight_layout()

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_ann_visualization_with_masks(img: Image.Image, det_entries: List[Dict], out_path: str) -> None:
    """
    Сохраняет визуализацию детекций с bounding boxes и масками.
    
    Создаёт изображение с наложенными bbox и полупрозрачными масками сегментации,
    используя цветовое кодирование по классам и легенду.
    
    Параметры:
        img (PIL.Image): исходное изображение
        det_entries (List[Dict]): список детекций с полями 'bbox_xyxy', 'class', 'mask', 'score_metric'
        out_path (str): путь для сохранения визуализации
        
    Возвращает:
        None
    """
    class_names = [d.get('class', 'unknown') for d in det_entries]
    unique_classes = list(dict.fromkeys(class_names))
    colors = plt.cm.Set3(np.linspace(0, 1, max(len(unique_classes), 1)))
    class_to_color = {cls: colors[i % len(colors)] for i, cls in enumerate(unique_classes)}

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.imshow(np.array(img))

    # Строим единый RGBA-оверлей для всех масок и рисуем его один раз
    H, W = img.height, img.width
    overlay = np.zeros((H, W, 4), dtype=np.uint8)
    alpha_val = int(0.4 * 255)
    for d in det_entries:
        mask = d.get('mask')
        if mask is None:
            continue
        color = class_to_color.get(d.get('class', 'unknown'), (1.0, 0.0, 0.0, 1.0))
        rgb255 = (np.array(color[:3]) * 255).astype(np.uint8)
        overlay[mask] = [*rgb255, 0] # полупрозрачная маска
        # максимизируем альфу там, где перекрытие
        overlay_alpha_view = overlay[..., 3]
        overlay_alpha_view[mask] = np.maximum(overlay_alpha_view[mask], alpha_val)
    if np.any(overlay[..., 3] > 0):
        ax.imshow(overlay, interpolation='nearest')

    # Затем рисуем все bbox и текст поверх
    for d in det_entries:
        x1,y1,x2,y2 = d["bbox_xyxy"]
        w = max(0.0, float(x2) - float(x1))
        h = max(0.0, float(y2) - float(y1))
        cls_name = d.get('class', 'unknown')
        score = d.get('score_metric', None)
        color = class_to_color.get(cls_name, (1.0, 0.0, 0.0, 1.0))

        # Рисуем bbox
        rect = patches.Rectangle((x1, y1), w, h, linewidth=3,
                                 edgecolor=color, facecolor='none', alpha=0.9)
        ax.add_patch(rect)

        label = cls_name if (score is None or not (score==score and math.isfinite(score))) else f"{cls_name} {score:.2f}"
        ax.text(x1, max(0, y1 - 10), label, fontsize=8, color='white',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.85),
                fontweight='bold')

    ax.set_xlim(0, img.width)
    ax.set_ylim(img.height, 0)
    ax.axis('off')

    # Легенда по классам (цвет -> класс), чтобы цвета были понятны
    if unique_classes:
        legend_elements = [patches.Patch(color=class_to_color[c], label=c) for c in unique_classes]
        # Размещаем легенду справа сверху, чуть наружу поля изображения
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=10)
    plt.tight_layout()

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


class ExperimentLogger:
    """
    Логгер для записи результатов эксперимента аннотации.
    
    Управляет сохранением всех артефактов эксперимента: CSV с метриками времени,
    JSON с аннотациями, копирование исходных изображений и визуализации.
    Поддерживает асинхронное сохранение для ускорения работы.
    
    Параметры:
        exp_dir (str): директория эксперимента для сохранения всех артефактов
        
    Методы:
        log_timing(rec: dict): добавляет запись времени этапов в CSV
        save_json(results): сохраняет аннотации в JSON (native или COCO формат)
        save_json_async(results): асинхронное сохранение JSON
        wait_json(timeout): ожидание завершения асинхронного сохранения
        save_vis(img, name): сохраняет визуализацию
        copy_source(path): копирует исходное изображение в папку эксперимента
    """

    def __init__(self, exp_dir: str):
        self.exp_dir = exp_dir
        self.csv_path = os.path.join(exp_dir, "timings.csv")
        self.json_path = os.path.join(exp_dir, "predictions.json")
        self._csv_initialized = False
        self._json_thread = None

    def log_timing(self, rec: Dict) -> None:
        """
        Записывает метрики времени выполнения этапов в CSV файл.
        
        Параметры:
            rec (Dict): словарь с метриками (image, num_dets, time_total, time_det, time_sam_total, time_embed_total)
            
        Возвращает:
            None
        """
        headers = [
            "image", "num_dets", "time_total", "time_det", "time_sam_total", "time_embed_total"
        ]
        write_header = not self._csv_initialized and not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            if write_header:
                w.writeheader()
            w.writerow({k: rec.get(k, "") for k in headers})
        self._csv_initialized = True

    @staticmethod
    def _get_image_dimensions(file_name: str, images_dir: str) -> Tuple[int, int]:
        width, height = 0, 0
        if file_name:
            img_path = os.path.join(images_dir, file_name)
            try:
                with Image.open(img_path) as _im:
                    width, height = _im.size
            except Exception:
                pass
        return width, height

    def save_json(self, results: List[Dict], target_path: str | None = None) -> None:
        """
        Сохраняет результаты аннотации в JSON файл.
        
        Поддерживает два формата: native (оригинальная структура) и COCO
        (стандартный формат для компьютерного зрения). Маски кодируются
        в RLE формат для компактности.
        
        Параметры:
            results (List[Dict]): список результатов обработки изображений
            
        Возвращает:
            None
        """
        out_path = target_path or self.json_path
        # Ветка Label Studio: быстрый путь, без тяжёлой санации масок/RLE
        if OUTPUT_FORMAT == "label_studio":
            def _build_prediction(rec: Dict) -> Dict | None:
                file_name = rec.get('image')
                width = rec.get('width') or 0
                height = rec.get('height') or 0
                if width == 0 or height == 0:
                    w, h = self._get_image_dimensions(file_name, IMAGES_DIR)
                    width = width or w
                    height = height or h
                if width == 0 or height == 0:
                    try:
                        print(f"[WARN] Пропуск изображения {file_name}: не удалось определить размеры.")
                    except Exception:
                        pass
                    return None
                ls_result = []
                for det in rec.get('detections', []) or []:
                    x1, y1, x2, y2 = det.get('bbox_xyxy', [0, 0, 0, 0])
                    x = (float(x1) / float(width)) * 100.0 if width else 0.0
                    y = (float(y1) / float(height)) * 100.0 if height else 0.0
                    w = ((float(x2) - float(x1)) / float(width)) * 100.0 if width else 0.0
                    h = ((float(y2) - float(y1)) / float(height)) * 100.0 if height else 0.0
                    cls_name = det.get('class', 'unknown') or 'unknown'
                    score = det.get('score_metric', 0.0) or 0.0
                    ls_result.append({
                        "id": str(uuid.uuid4()),
                        "from_name": "label",
                        "to_name": "img",
                        "type": "rectanglelabels",
                        "value": {"x": x, "y": y, "width": w, "height": h, "rectanglelabels": [cls_name]},
                        "score": float(score) if isinstance(score, (int, float)) and math.isfinite(float(score)) else 0.0,
                    })
                if not ls_result:
                    return None
                return {
                    "id": str(uuid.uuid4()),
                    "task": None,
                    "model_version": "dino_sam_dino3",
                    "score": max([res.get("score", 0.0) for res in ls_result]) if ls_result else 0.0,
                    "result": ls_result,
                    "image_filename": file_name,
                }

            # Параллельная сборка предсказаний
            if Parallel is not None and delayed is not None:
                n_jobs = max(1, min((os.cpu_count() or 1), 16))
                try:
                    print(f"[SAVE] Параллельная сборка предсказаний Label Studio: n_jobs={n_jobs}")
                except Exception:
                    pass
                preds = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(_build_prediction)(rec) for rec in (tqdm(results, desc="LS predictions", unit="img") if JSON_SAVE_TQDM else results))
                out_obj = [p for p in preds if p is not None]
            else:
                iterable_ls = tqdm(results, desc="LS predictions", unit="img") if JSON_SAVE_TQDM else results
                out_obj = []
                for rec in iterable_ls:
                    p = _build_prediction(rec)
                    if p is not None:
                        out_obj.append(p)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out_obj, f, ensure_ascii=False, indent=2)
            return

        # Ниже — общая тяжёлая санация (нужна для native/coco), с RLE масок
        # Подготовка результатов к сериализации: кодируем маски и чистим числа
        def mask_to_rle(mask: np.ndarray) -> Dict:
            # Ожидается булева маска HxW
            h, w = mask.shape
            flat = mask.reshape(-1).astype(np.uint8)
            counts: List[int] = []
            prev_val = 0  # Всегда начинаем с количества нулей
            run_len = 0
            for v in flat:
                if int(v) == prev_val:
                    run_len += 1
                else:
                    counts.append(int(run_len))
                    run_len = 1
                    prev_val = int(v)
            counts.append(int(run_len))
            return {"size": [int(h), int(w)], "counts": counts, "order": "row-major"}

        def rle_area(rle: Dict) -> float:
            counts = rle.get("counts", [])
            # при кодировке начиная с нулей площадь — сумма нечётных отрезков
            s = 0
            for i in range(1, len(counts), 2):
                try:
                    s += int(counts[i])
                except Exception:
                    pass
            return float(s)

        # Подготовка результатов к сериализации: удаляем несериализуемые поля и NaN
        def sanitize_det_entry(entry: Dict) -> Dict:
            e = dict(entry)
            # Кодируем маску в RLE
            m = e.pop('mask', None)
            if m is not None:
                try:
                    e['mask_rle'] = mask_to_rle(np.asarray(m, dtype=bool))
                except Exception:
                    e['mask_rle'] = None
            # score_metric делаем валидным числом или None
            sm = e.get('score_metric', None)
            if not (isinstance(sm, (int, float)) and math.isfinite(sm)):
                e['score_metric'] = None
            # Приведём ranked-список
            ranked = e.get('ranked', None)
            if isinstance(ranked, list):
                new_ranked = []
                for r in ranked:
                    rr = dict(r)
                    sc = rr.get('score', None)
                    if not (isinstance(sc, (int, float)) and math.isfinite(sc)):
                        rr['score'] = None
                    new_ranked.append(rr)
                e['ranked'] = new_ranked
            return e

        serializable_results: List[Dict] = []
        iterable = results
        if JSON_SAVE_TQDM:
            iterable = tqdm(results, desc="Saving JSON", unit="img")
        for rec in iterable:
            rec_out = dict(rec)
            dets = rec_out.get('detections', [])
            if isinstance(dets, list):
                rec_out['detections'] = [sanitize_det_entry(d) for d in dets]
            serializable_results.append(rec_out)

        if OUTPUT_FORMAT == "native":
            out_obj = serializable_results
        elif OUTPUT_FORMAT == "coco":
            # Построение COCO
            categories_map: Dict[str, int] = {}
            categories_list: List[Dict] = []
            def get_cat_id(name: str) -> int:
                if name not in categories_map:
                    categories_map[name] = len(categories_map) + 1
                    categories_list.append({"id": categories_map[name], "name": name})
                return categories_map[name]

            images_list: List[Dict] = []
            annotations_list: List[Dict] = []
            ann_id = 1
            for img_id, rec in enumerate(serializable_results, start=1):
                file_name = rec.get('image')
                width = int(rec.get('width', 0) or 0)
                height = int(rec.get('height', 0) or 0)
                # если нет размеров — попытаться прочитать
                if (width == 0 or height == 0) and file_name:
                    img_path = os.path.join(IMAGES_DIR, file_name)
                    try:
                        with Image.open(img_path) as _im:
                            width, height = _im.size
                    except Exception:
                        pass
                images_list.append({
                    "id": img_id,
                    "file_name": file_name,
                    "width": int(width),
                    "height": int(height),
                })
                for det in rec.get('detections', []) or []:
                    cls_name = det.get('class', 'unknown') or 'unknown'
                    cat_id = get_cat_id(cls_name)
                    x1,y1,x2,y2 = det.get('bbox_xyxy', [0,0,0,0])
                    x1 = float(x1); y1 = float(y1); x2 = float(x2); y2 = float(y2)
                    w = max(0.0, x2 - x1)
                    h = max(0.0, y2 - y1)
                    seg = det.get('mask_rle', None)
                    area = (w * h)
                    if seg is not None:
                        try:
                            area = rle_area(seg)
                        except Exception:
                            pass
                    annotations_list.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cat_id,
                        "bbox": [x1, y1, w, h],
                        "area": float(area),
                        "iscrowd": 0,
                        "segmentation": seg if seg is not None else [],
                    })
                    ann_id += 1
            out_obj = {
                "images": images_list,
                "annotations": annotations_list,
                "categories": categories_list,
            }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)

    def save_json_async(self, results: List[Dict]) -> None:
        def _worker():
            try:
                self.save_json(results)
                print(f"[OK] JSON сохранён (async): {self.json_path}")
            except Exception as e:
                print(f"[FAIL] Ошибка асинхронного сохранения JSON: {e}")
        # используем не-демонский поток, чтобы дождаться завершения при выходе
        t = Thread(target=_worker, daemon=False)
        t.start()
        self._json_thread = t
        print(f"[INFO] Асинхронное сохранение JSON запущено: {self.json_path}")

    def wait_json(self, timeout: float | None = None) -> None:
        try:
            if self._json_thread is not None and self._json_thread.is_alive():
                self._json_thread.join(timeout=timeout)
                if self._json_thread.is_alive():
                    print("[WARN] Ожидание сохранения JSON превысило таймаут")
        except Exception:
            pass

    def save_vis(self, img: Image.Image, name: str) -> None:
        img.save(os.path.join(self.exp_dir, name))

    def copy_source(self, path: str) -> None:
        dst = os.path.join(self.exp_dir, os.path.basename(path))
        try:
            if os.path.abspath(path) != os.path.abspath(dst):
                shutil.copy2(path, dst)
        except Exception as e:
            print("copy_source fail", path, e)


def init_embedder(device: torch.device) -> DinoV3Embedder:
    """
    Инициализирует эмбеддер DINOv3 с автоматическим выбором режима.
    
    Пытается загрузить обученный чекпойнт EmbedNet, при неудаче
    переключается на vanilla DINOv3 из timm. Поддерживает мульти-GPU
    режим при наличии второго устройства.
    
    Параметры:
        device (torch.device): основное устройство для эмбеддера
        
    Возвращает:
        DinoV3Embedder: инициализированный эмбеддер (может быть разных типов)
    """
    # Управление режимом по флагам
    if DINO_USE_MODEL_PARALLEL and (dino_mp_second_device is not None) and (device.type == "cuda"):
        print(f"[INIT] Инициализация DINOv3 (2GPU MP=True) на {device}+{dino_mp_second_device}...")
        try:
            emb = DinoV3Embedder2GPU(TIMM_DINOV3_ID, device, dino_mp_second_device)
            print("[OK] DINOv3 шардирован на 2 GPU")
            return emb
        except Exception as e:
            print("[WARN] Не удалось инициализировать 2-GPU режим:", e)
            print("[FALLBACK] Переход на одно устройство.")
    if DINO3_USE_CHECKPOINT and os.path.isfile(DINO3_CKPT):
        print(f"[INIT] Загрузка обученного EmbedNet из чекпойнта: {DINO3_CKPT}")
        try:
            model = EmbedNet(dim=DINO3_EMB_DIM, model_name=DINO3_MODEL_ID, freeze_backbone=True, img_size=DINO3_IMG_SIZE).to(device).eval()
            ckpt = torch.load(DINO3_CKPT, map_location=device)
            state_dict = ckpt.get("model") or ckpt
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                print("[WARN] State dict diff:", {"missing": missing, "unexpected": unexpected})
            # Собираем transforms как в тренировке: через timm data config
            tmp_model = timm.create_model(DINO3_MODEL_ID, pretrained=True, num_classes=0, global_pool="token")
            data_cfg = timm.data.resolve_model_data_config(tmp_model)
            data_cfg["input_size"] = (3, DINO3_IMG_SIZE, DINO3_IMG_SIZE)
            transforms = timm.data.create_transform(**data_cfg, is_training=False)
            class _Wrap:
                def __init__(self, model, transforms, device):
                    self.model = model
                    self.transforms = transforms
                    self.device = device
                @torch.no_grad()
                def embed(self, img: Image.Image) -> torch.Tensor:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    t_img = self.transforms(img).unsqueeze(0).to(self.device)
                    z = self.model(t_img)
                    return z.squeeze(0).detach().cpu().float()
            emb = _Wrap(model, transforms, device)
            print(f"[OK] EmbedNet из чекпойнта готов на {device}")
            return emb
        except Exception as e:
            print("[FAIL] Не удалось загрузить EmbedNet чекпойнт:", e)
            print("[FALLBACK] Переход на DINOv3 (timm)")
    print(f"[INIT] Инициализация DINOv3 эмбеддера (timm, single-device, fp16={DINO_SINGLE_GPU_FP16}) на {device}...")
    try:
        emb = DinoV3Embedder(TIMM_DINOV3_ID, device, use_fp16=(device.type=="cuda" and DINO_SINGLE_GPU_FP16))
        print(f"[OK] DINOv3 (timm) инициализирован на {device}")
        return emb
    except Exception as e:
        print("[FAIL] DINOv3 не инициализирован:", e)
        raise


def init_detector(device: torch.device) -> GroundingDINODetector:
    """
    Инициализирует детектор GroundingDINO.
    
    Создаёт экземпляр детектора с предустановленными параметрами
    для поиска инструментов по текстовому промпту.
    
    Параметры:
        device (torch.device): устройство для размещения модели
        
    Возвращает:
        GroundingDINODetector: инициализированный детектор
    """
    print(f"[INIT] Инициализация GroundingDINO: {GROUNDING_DINO_MODEL} на {device}...")
    try:
        det = GroundingDINODetector(
            model_name=GROUNDING_DINO_MODEL,
            prompt=GROUNDING_PROMPT,
            box_thr=BOX_THR,
            text_thr=TEXT_THR,
            device=device,
        )
        print(f"[OK] GroundingDINO инициализирован на {device}")
        return det
    except Exception as e:
        print(f"[FAIL] GroundingDINO не инициализирован на {device}:", e)
        raise


def init_segmenter(device: torch.device) -> SAM2Segmenter:
    """
    Инициализирует сегментатор SAM-2.
    
    Создаёт экземпляр SAM-2 для генерации масок сегментации
    по bounding boxes, найденным детектором.
    
    Параметры:
        device (torch.device): устройство для размещения модели
        
    Возвращает:
        SAM2Segmenter: инициализированный сегментатор
    """
    print(f"[INIT] Инициализация SAM-2 (large) на {device}...")
    try:
        seg = SAM2Segmenter(SAM2_CFG, SAM2_CKPT, device)
        print(f"[OK] SAM-2 инициализирован на {device}")
        return seg
    except Exception as e:
        print(f"[FAIL] SAM-2 не инициализирован на {device}:", e)
        raise


@measure_time
def run_detection(detector: GroundingDINODetector, img: Image.Image) -> List[Dict]:
    """
    Выполняет детекцию объектов на изображении с замером времени.
    
    Параметры:
        detector (GroundingDINODetector): инициализированный детектор
        img (PIL.Image): изображение для обработки
        
    Возвращает:
        List[Dict]: список детекций с bbox и скорами (время возвращается декоратором)
    """
    return detector.detect(img)


@measure_time
def run_sam_for_detections(segmenter: SAM2Segmenter, img_rgb: Image.Image, dets: List[Dict]) -> List[Dict]:
    """
    Генерирует маски сегментации для найденных bounding boxes с замером времени.
    
    Для каждого bbox из детектора запускает SAM-2 для получения точной маски
    и вычисляет tight bounding box по полученной маске.
    
    Параметры:
        segmenter (SAM2Segmenter): инициализированный сегментатор SAM-2
        img_rgb (PIL.Image): RGB изображение для обработки  
        dets (List[Dict]): список детекций с bbox_xyxy от GroundingDINO
        
    Возвращает:
        List[Dict]: список словарей с полями {mask, tight, bbox, score_vlm, vlm_class}
                   (время возвращается декоратором)
    """
    img_np = np.array(img_rgb)
    segmenter.set_image(img_np)
    out = []
    for idx, d in enumerate(dets):
        x1,y1,x2,y2 = d["bbox_xyxy"]
        m = segmenter.mask_from_box([x1,y1,x2,y2])
        tight = SAM2Segmenter.tight_bbox_from_mask(m)
        # Если включена отладка, сохраним маску отдельно позже, внутри process_image
        out.append({
            "mask": m,
            "tight": tight,
            "bbox": [x1,y1,x2,y2],
            "score_vlm": d.get("score_vlm", 0.0),
            "vlm_class": d.get("class", "")
        })
    return out


@measure_time
def run_embed_and_match(embedder: DinoV3Embedder, gallery_names: List[str], gallery_vecs: torch.Tensor, img_rgb: Image.Image, sam_outputs: List[Dict], debug_dir: str | None = None, gallery_paths: List[str] | None = None) -> List[Dict]:
    """
    Извлекает эмбеддинги и сопоставляет с галереей эталонов с замером времени.
    
    Для каждой найденной маски вырезает объект, извлекает эмбеддинг и находит
    наиболее похожий эталон из галереи. Создаёт ранжированный список похожести
    для каждого объекта и опционально экспортирует отладочные артефакты.
    
    Параметры:
        embedder (DinoV3Embedder): инициализированный эмбеддер
        gallery_names (List[str]): имена эталонов из галереи
        gallery_vecs (torch.Tensor): матрица эмбеддингов галереи [N, D]
        img_rgb (PIL.Image): исходное RGB изображение
        sam_outputs (List[Dict]): результаты SAM-2 с масками
        debug_dir (str | None): папка для сохранения отладочных артефактов
        gallery_paths (List[str] | None): пути к файлам галереи для отладки
        
    Возвращает:
        List[Dict]: список детекций с классификацией и ранжированием
                   (время возвращается декоратором)
    """
    det_entries = []
    for idx, item in enumerate(sam_outputs):
        cut = SAM2Segmenter.apply_mask_and_crop(img_rgb, item["mask"])
        z = embedder.embed(cut)
        # Всегда считаем похожести по всей галерее для ранжирования
        with torch.no_grad():
            z32 = torch.nan_to_num(z.float(), nan=0.0, posinf=0.0, neginf=0.0)
            z32 = F.normalize(z32.view(-1), dim=0, eps=1e-12)
            G32 = torch.nan_to_num(gallery_vecs.float(), nan=0.0, posinf=0.0, neginf=0.0)
            G32 = F.normalize(G32, dim=1, eps=1e-12)
            sims_all = torch.mv(G32, z32)
        top_idx = int(torch.argmax(sims_all).item())
        name = _canon_class(gallery_names[top_idx])
        score = float(sims_all[top_idx].item())
        # Ранжированный список по убыванию
        sorted_idx = torch.argsort(sims_all, descending=True).tolist()
        ranked = [{"name": _canon_class(gallery_names[j]), "score": float(sims_all[j].item())} for j in sorted_idx]
        # Лог о классе от DinoV3
        try:
            print(f"    [EMB] Маска/кроп -> класс: {name} со скором {score:.4f}")
        except Exception:
            pass
        # Отладочный вывод: сохранение кропа и сравнений
        if debug_dir is not None:
            os.makedirs(debug_dir, exist_ok=True)
            crop_dir = os.path.join(debug_dir, f"crop_{idx:02d}")
            os.makedirs(crop_dir, exist_ok=True)
            # Сохраняем кроп и маску
            try:
                cut_path = os.path.join(crop_dir, "crop.png")
                cut.save(cut_path)
            except Exception:
                pass
            # Сохраняем маску как изображение
            try:
                mask_img = Image.fromarray((item["mask"].astype(np.uint8) * 255))
                mask_path = os.path.join(crop_dir, "mask.png")
                mask_img.save(mask_path)
            except Exception:
                pass
            # Сохраняем карту похожести по всем эталонам с корректным путём
            # Строим имя -> индекс в галерее (предполагаем уникальные имена эталонов)
            name_to_idx = None
            if gallery_paths is not None:
                name_to_idx = {_canon_class(os.path.splitext(os.path.basename(p))[0]): j for j, p in enumerate(gallery_paths)}
            sims_json = []
            for r in ranked:
                rname = _canon_class(r.get("name"))
                ridx = None
                rpath = None
                if name_to_idx is not None and rname in name_to_idx:
                    ridx = name_to_idx.get(rname)
                    try:
                        rpath = gallery_paths[ridx]
                    except Exception:
                        rpath = None
                sims_json.append({"name": rname, "score": r.get("score"), "path": rpath})
            try:
                best_path = None
                if name_to_idx is not None and _canon_class(name) in name_to_idx:
                    try:
                        best_path = gallery_paths[name_to_idx[_canon_class(name)]]
                    except Exception:
                        best_path = None
                with open(os.path.join(crop_dir, "similarities.json"), "w", encoding="utf-8") as f:
                    json.dump({"best": {"name": name, "score": float(score), "path": best_path}, "all": sims_json}, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            # Скопировать эталонные изображения для визуальной проверки
            if gallery_paths:
                gal_dir = os.path.join(crop_dir, "gallery")
                os.makedirs(gal_dir, exist_ok=True)
                for i, gp in enumerate(gallery_paths):
                    try:
                        dst = os.path.join(gal_dir, f"{i:02d}_{os.path.basename(gp)}")
                        if not os.path.exists(dst):
                            shutil.copy2(gp, dst)
                    except Exception:
                        pass
        det_entries.append({
            "bbox_xyxy": item["tight"],
            "class": name,
            "score_metric": float(score) if score == score and math.isfinite(score) else float("nan"),
            "score_vlm": float(item["score_vlm"]),
            "vlm_class": item["vlm_class"],
            "ranked": ranked,
            "mask": item["mask"],  # добавляем маску для визуализации
        })
    return det_entries


def process_image(img_path: str, embedder: DinoV3Embedder, detector: GroundingDINODetector, segmenter: SAM2Segmenter, gallery_names: List[str], gallery_vecs: torch.Tensor, logger: ExperimentLogger, gallery_paths: List[str] | None = None) -> Dict:
    """
    Выполняет полный цикл обработки одного изображения.
    
    Последовательно применяет детекцию, сегментацию, эмбеддинг и классификацию.
    Сохраняет визуализации, метрики времени и отладочные артефакты.
    
    Параметры:
        img_path (str): путь к изображению для обработки
        embedder (DinoV3Embedder): эмбеддер для извлечения признаков
        detector (GroundingDINODetector): детектор для поиска объектов
        segmenter (SAM2Segmenter): сегментатор для генерации масок
        gallery_names (List[str]): имена эталонов из галереи
        gallery_vecs (torch.Tensor): матрица эмбеддингов галереи
        logger (ExperimentLogger): логгер для сохранения результатов
        gallery_paths (List[str] | None): пути к файлам галереи для отладки
        
    Возвращает:
        Dict: результат обработки с полями image, width, height, detections
    """
    stem = os.path.splitext(os.path.basename(img_path))[0]
    print(f"[IMG] Обработка: {stem}")
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print("[WARN] Пропуск (не открыть):", img_path, e)
        return {"image": os.path.basename(img_path), "detections": [], "error": str(e)}

    logger.copy_source(img_path)

    (dets, t_det) = run_detection(detector, img)
    print(f"  - GroundingDINO: {len(dets)} dets за {t_det:.3f}s")

    (sam_outputs, t_sam) = run_sam_for_detections(segmenter, img, dets)
    print(f"  - SAM-2: готово за {t_sam:.3f}s")

    # Подготовка debug директорий
    per_img_debug_dir = None
    if DEBUG_EXPORT:
        per_img_debug_dir = os.path.join(EXP_DIR, "debug", stem)
        os.makedirs(per_img_debug_dir, exist_ok=True)
        # Сохранить исходное изображение
        try:
            img.save(os.path.join(per_img_debug_dir, "source.jpg"))
        except Exception:
            pass

    (det_entries, t_emb) = run_embed_and_match(embedder, gallery_names, gallery_vecs, img, sam_outputs, debug_dir=per_img_debug_dir, gallery_paths=gallery_paths)
    print(f"  - DinoV3: сопоставление за {t_emb:.3f}s")

    # Визуализация Matplotlib (до перераспределения, только bbox)
    if SAVE_VIS:
        save_ann_visualization_matplotlib(img, det_entries, os.path.join(logger.exp_dir, f"{stem}_ann.jpg"))

    # Перераспределение классов до уникальности и сохранение _recalculated
    det_entries_unique = det_entries
    try:
        det_entries_unique = assign_unique_classes(det_entries)
        if SAVE_VIS:
            # Визуализация после перераспределения (только bbox)
            save_ann_visualization_matplotlib(img, det_entries_unique, os.path.join(logger.exp_dir, f"{stem}_ann_recalculated.jpg"))
            # Визуализация после перераспределения (bbox + маски)
            save_ann_visualization_with_masks(img, det_entries_unique, os.path.join(logger.exp_dir, f"{stem}_ann_with_masks.jpg"))
        # Информативный лог о разнообразии классов и количестве боксов
        try:
            uniq_names = list(dict.fromkeys([d.get("class", "") for d in det_entries_unique]))
            print(f"  - Recalc: {len(det_entries_unique)} dets, уникальных классов: {len(uniq_names)} -> {uniq_names}")
            # Доп.проверка уникальности; логируем предупреждение, если есть дубли
            if len(uniq_names) != len(det_entries_unique):
                print("[WARN] После перерасчёта остались дубли классов; проверьте входные 'ranked' и имена эталонов")
        except Exception:
            pass
    except Exception as e:
        print("[WARN] Не удалось перераспределить классы уникально:", e)

    total_time = t_det + t_sam + t_emb
    logger.log_timing({
        "image": os.path.basename(img_path),
        "num_dets": len(dets),
        "time_total": f"{total_time:.4f}",
        "time_det": f"{t_det:.4f}",
        "time_sam_total": f"{t_sam:.4f}",
        "time_embed_total": f"{t_emb:.4f}",
    })

    # В JSON отдаём детекции ДО перераспределения (соответствуют {stem}_ann.jpg)
    return {"image": os.path.basename(img_path), "width": int(img.width), "height": int(img.height), "detections": det_entries}


def annotate_images(paths: List[str], embedder: DinoV3Embedder, detector: GroundingDINODetector, segmenter: SAM2Segmenter, gallery_names: List[str], gallery_vecs: torch.Tensor, logger: ExperimentLogger) -> List[Dict]:
    """
    Аннотирует батч изображений с прогресс-баром.
    
    Последовательно обрабатывает список изображений, применяя полный пайплайн
    детекции, сегментации и классификации. Показывает прогресс выполнения.
    
    Параметры:
        paths (List[str]): список путей к изображениям для обработки
        embedder (DinoV3Embedder): эмбеддер для извлечения признаков
        detector (GroundingDINODetector): детектор для поиска объектов
        segmenter (SAM2Segmenter): сегментатор для генерации масок
        gallery_names (List[str]): имена эталонов из галереи
        gallery_vecs (torch.Tensor): матрица эмбеддингов галереи
        logger (ExperimentLogger): логгер для сохранения результатов
        
    Возвращает:
        List[Dict]: список результатов обработки для каждого изображения
    """
    results = []
    # Резюмирование: загружаем все существующие JSON файлы (основной + части + fixed)
    processed_set: set[str] = set()
    exp_dir = os.path.dirname(logger.json_path)
    
    # Загружаем основной файл
    try:
        if os.path.isfile(logger.json_path):
            with open(logger.json_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            for rec in existing:
                fn = rec.get("image") or rec.get("image_filename")
                if fn:
                    processed_set.add(str(fn))
            try:
                print(f"[RESUME] Основной файл: {len(existing)} записей")
            except Exception:
                pass
    except Exception as e:
        try:
            print(f"[RESUME] Не удалось загрузить основной predictions.json: {e}")
        except Exception:
            pass
    
    # Загружаем все части
    part_files = sorted(glob.glob(os.path.join(exp_dir, "predictions_part*.json")))
    for part_file in part_files:
        try:
            with open(part_file, "r", encoding="utf-8") as f:
                part_data = json.load(f)
            for rec in part_data:
                fn = rec.get("image") or rec.get("image_filename")
                if fn and fn not in processed_set:  # избегаем дублей
                    processed_set.add(str(fn))
            try:
                print(f"[RESUME] Часть {os.path.basename(part_file)}: +{len(part_data)} записей")
            except Exception:
                pass
        except Exception as e:
            try:
                print(f"[RESUME] Ошибка в части {part_file}: {e}")
            except Exception:
                pass

    # Учитываем predictions_fixed.json, если есть
    fixed_path = os.path.join(exp_dir, "predictions_fixed.json")
    try:
        if os.path.isfile(fixed_path):
            with open(fixed_path, "r", encoding="utf-8") as f:
                fixed_data = json.load(f)
            for rec in fixed_data:
                fn = rec.get("image") or rec.get("image_filename")
                if fn and fn not in processed_set:
                    processed_set.add(str(fn))
            try:
                print(f"[RESUME] Файл fixed: +{len(fixed_data)} записей")
            except Exception:
                pass
    except Exception as e:
        try:
            print(f"[RESUME] Ошибка чтения predictions_fixed.json: {e}")
        except Exception:
            pass
    
    try:
        print(f"[RESUME] Итого загружено: {len(processed_set)} изображений, продолжу с оставшихся")
    except Exception:
        pass
    
    # Фильтруем пути: оставляем только необработанные
    remaining_paths = []
    for p in paths:
        stem = os.path.basename(p)
        if stem not in processed_set:
            remaining_paths.append(p)
    
    try:
        print(f"[INFO] Всего файлов: {len(paths)}, уже обработано: {len(processed_set)}, осталось: {len(remaining_paths)}")
    except Exception:
        pass
    
    # Определяем следующий номер part-файла на основе существующих
    next_part_num = 1
    try:
        import re
        nums = []
        for pf in part_files:
            m = re.search(r"predictions_part(\d+)\.json$", os.path.basename(pf))
            if m:
                try:
                    nums.append(int(m.group(1)))
                except Exception:
                    pass
        if nums:
            next_part_num = max(nums) + 1
    except Exception:
        pass

    # Индекс последней сохранённой записи в текущем забеге
    last_saved_idx = 0

    pbar = tqdm(remaining_paths, desc="Annotating", unit="img")
    # Если включён DEBUG_EXPORT — создадим корневую папку
    if DEBUG_EXPORT:
        os.makedirs(os.path.join(EXP_DIR, "debug"), exist_ok=True)
    for p in pbar:
        pbar.set_postfix(stage="start")
        
        # Мониторинг ресурсов каждые 10 изображений
        if len(results) % 10 == 0:
            try:
                mem_percent = psutil.virtual_memory().percent
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.memory_allocated() / 1024**3
                    pbar.set_postfix(stage=f"RAM:{mem_percent:.1f}% GPU:{gpu_mem:.1f}GB")
                else:
                    pbar.set_postfix(stage=f"RAM:{mem_percent:.1f}%")
            except Exception:
                pass
        
        # Передаём пути галереи, если известны
        rec = process_image(p, embedder, detector, segmenter, gallery_names, gallery_vecs, logger, gallery_paths=getattr(annotate_images, "_gallery_paths", None))
        results.append(rec)
        
        # Принудительная очистка памяти каждые 25 изображений
        if len(results) % 25 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Промежуточное сохранение результатов каждые JSON_CHECKPOINT_EVERY изображений
        if JSON_CHECKPOINT_EVERY and (len(results) % int(JSON_CHECKPOINT_EVERY) == 0):
            try:
                # Инкрементально пишем новую часть, не перезаписывая существующие
                part_path = logger.json_path.replace('.json', f'_part{next_part_num:03d}.json')
                current_part = results[last_saved_idx:]
                if current_part:
                    logger.save_json(current_part, target_path=part_path)
                    last_saved_idx = len(results)
                    try:
                        print(f"[CKPT] Новая часть {next_part_num}: {len(current_part)} записей -> {part_path}")
                    except Exception:
                        pass
                    next_part_num += 1
            except Exception as e:
                try:
                    print(f"[WARN] Ошибка сохранения JSON на {len(results)}: {e}")
                except Exception:
                    pass
    # Сохраняем хвост (если остался) отдельной частью
    try:
        if last_saved_idx < len(results):
            part_path = logger.json_path.replace('.json', f'_part{next_part_num:03d}.json')
            current_part = results[last_saved_idx:]
            if current_part:
                logger.save_json(current_part, target_path=part_path)
                try:
                    print(f"[CKPT] Финальная часть {next_part_num}: {len(current_part)} записей -> {part_path}")
                except Exception:
                    pass
    except Exception as e:
        try:
            print(f"[WARN] Ошибка финального сохранения части: {e}")
        except Exception:
            pass
    return results

def main():
    """
    Главная функция для запуска пайплайна автоматической аннотации.
    
    Инициализирует все модели, загружает галерею эталонов, обрабатывает
    изображения и сохраняет результаты в JSON и визуализации.
    
    Пример запуска:
        # Настройте константы в начале файла (IMAGES_DIR, GALLERY_DIR, BASE_DIR)
        # и запустите:
        python annotate_gdino_sam2_dinov3_hf.py
        
        # Результаты будут сохранены в BASE_DIR:
        # - predictions.json: аннотации в native или COCO формате
        # - timings.csv: метрики времени выполнения этапов
        # - *_ann.jpg: визуализации с bbox
        # - *_ann_recalculated.jpg: визуализации после перераспределения классов
        # - *_ann_with_masks.jpg: визуализации с масками
        # - debug/: отладочные артефакты (при DEBUG_EXPORT=True)
    
    Возвращает:
        None
    """
    print("[MAIN] Запуск пайплайна разметки")
    logger = ExperimentLogger(EXP_DIR)
    try:
        print(f"[OUT] Результаты будут сохранены в: {EXP_DIR}\n      JSON: {logger.json_path}")
    except Exception:
        pass

    embedder = init_embedder(device_embed)
    print("[STAGE] Загрузка галереи эталонов...")
    gallery_names, gallery_vecs, gallery_paths = load_gallery(embedder, GALLERY_DIR)
    print(f"[OK] Галерея загружена: {len(gallery_names)} эталонов")
    # Сохраняем пути галереи в функцию аннотирования для debug-пайплайна
    setattr(annotate_images, "_gallery_paths", gallery_paths)

    detector = init_detector(device_det)
    segmenter = init_segmenter(device_seg)

    print("[STAGE] Поиск изображений...")
    all_imgs = sorted([p for p in glob.glob(os.path.join(IMAGES_DIR, "*")) if p.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp"))])[:NUM_IMAGES]
    print(f"[OK] Найдено изображений: {len(all_imgs)}")

    print("[STAGE] Аннотирование...")
    results = annotate_images(all_imgs, embedder, detector, segmenter, gallery_names, gallery_vecs, logger)
    # Не перезаписываем основной predictions.json, части уже сохранены в annotate_images
    print("✅ Разметка готова (части сохранены в EXP_DIR):", EXP_DIR)


if __name__ == "__main__":
    main()


