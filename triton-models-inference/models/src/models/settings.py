"""
Файл: /home/ubuntu/triton-models-inference/models/src/models/settings.py

Основной смысл: Централизованное хранение всех конфигурационных параметров для системы детекции объектов.
Содержит настройки для всех используемых моделей: GroundingDINO, DINOv3, а также пути к данным и параметры отладки.

Конфигурация включает:
- Параметры устройства (CPU/GPU)
- Настройки модели GroundingDINO для детекции
- Параметры модели DINOv3 для эмбеддингов
- Пути к чекпойнтам и галерее эталонов
- Режимы отладки

Использование:
    from models.settings import DEVICE, GROUNDING_DINO_MODEL, DINO3_CKPT
    
    # Использование в коде
    model = YOLO(GROUNDING_DINO_MODEL).to(DEVICE)
"""

import os
import torch

# Устройство инференса
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# GroundingDINO (детектор боксов)
GROUNDING_DINO_MODEL = "rziga/mm_grounding_dino_large_all"
GROUNDING_PROMPT = "tool"
BOX_THR = 0.25
TEXT_THR = 0.25

# DINOv3 (эмбеддер) — как в annotate_gdino_sam2_dinov3_hf.py
TIMM_DINOV3_ID = "vit_large_patch16_dinov3"

# Использовать обученный чекпойнт EmbedNet поверх базовой модели
DINO3_USE_CHECKPOINT = True
# Контейнерный путь к чекпойнту (см. docker compose volume)
DINO3_CKPT = "/data/checkpoints/dino3/checkpoint_last.pth"
DINO3_EMB_DIM = 128
# Базовая модель timm — как в annotate_gdino_sam2_dinov3_hf.py
DINO3_MODEL_ID = "vit_large_patch16_dinov3"
DINO3_IMG_SIZE = 224

# Путь к галерее эталонов (контейнер)
GALLERY_DIR = "/data/gallery"

# Debug / mock mode (enabled by default; override with env OBJECT_DETECTOR_DEBUG=0)
DEBUG_MOCK = (str(os.getenv("OBJECT_DETECTOR_DEBUG", "1")).strip() in ("1", "true", "yes", "y", "on"))

