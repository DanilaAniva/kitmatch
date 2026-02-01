"""
Обучение эмбеддера DINOv3 с ArcFace для классификации инструментов

Этот скрипт обучает нейронную сеть для извлечения признаков (эмбеддингов) из изображений 
инструментов. Использует предобученный DINOv3 в качестве backbone с дополнительной головой
и ArcFace margin loss для метрического обучения.

Основная логика:
1. Загружает обрезанные изображения инструментов из подпапок "cropped" каждого класса
2. Опционально использует маски для фокусировки на объекте (режимы: bbox/mask/mix)
3. Обучает эмбеддер с ArcFace loss для получения дискриминативных признаков
4. Поддерживает класс-балансную выборку и различные аугментации
5. Экспортирует прототипы (центроиды) классов для последующего поиска
6. Интегрируется с ClearML для трекинга экспериментов

Модели и технологии:
- DINOv3 (vit_large_patch16_dinov3.lvd1689m) как feature extractor
- ArcFace margin loss для метрического обучения
- Albumentations для аугментаций
- SAM-2 для автогенерации масок (опционально)
- ClearML для логирования экспериментов
"""

import os, sys, json, math, random, argparse, time, re
from typing import Dict, List, Tuple, Any

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import timm
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_fscore_support


MERGE_MAP = {
    # Объединение трех классов отверток в один "Отвертка"
    "1 Отвертка «-»": "Отвертка",
    "2 Отвертка «+»": "Отвертка",
    "3 Отвертка на смещенный крест": "Отвертка",
}

# Исключаем известные проблемные исходники по диапазонам DSCN-номеров (в любых классах)
# Диапазоны включительно
EXCLUDE_DSCN_RANGES: List[Tuple[int, int]] = [
    (4680, 4720),
    (4858, 4884),
]


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
    log = logging.getLogger("train")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(out_dir, "train.log"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch.setFormatter(fmt); fh.setFormatter(fmt)
    log.addHandler(ch); log.addHandler(fh)
    return log


def set_seed(seed: int = 42):
    """
    Устанавливает сид для воспроизводимости результатов.
    
    Args:
        seed: Значение сида
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_str(s: str) -> str:
    """
    Нормализует строку для сравнения.
    
    Args:
        s: Исходная строка
        
    Returns:
        Нормализованная строка (нижний регистр, без пробелов)
    """
    try:
        import unicodedata as ud
        return ud.normalize("NFKC", s).strip().lower()
    except Exception:
        return s.strip().lower()


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    """
    Безопасно дописывает одну JSON-строку в файл метрик.
    
    Используется для логирования метрик обучения в JSONL формате.
    
    Args:
        path: Путь к JSONL файлу
        record: Словарь с метриками для сохранения
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")
    except Exception:
        # Не прерываем обучение из-за проблем с логированием
        pass

def find_mask_for(img_path: str, masks_dirname: str = "masks", cutouts_dirname: str = "cutouts") -> np.ndarray | None:
    """
    Поиск маски для изображения по стандартным правилам.
    
    Ищет маску сначала в cutouts (через альфа-канал RGBA), 
    затем в masks (бинарные PNG маски).
    
    Args:
        img_path: Путь к изображению (обычно в подпапке "cropped")
        masks_dirname: Название папки с масками
        cutouts_dirname: Название папки с RGBA cutouts
        
    Returns:
        Булева маска [H, W] или None, если не найдена
    """
    try:
        class_dir = os.path.dirname(os.path.dirname(img_path))
        stem = os.path.splitext(os.path.basename(img_path))[0]
        cut_dir = os.path.join(class_dir, cutouts_dirname)
        for cp in [os.path.join(cut_dir, stem + "_cutout.png"), os.path.join(cut_dir, stem + ".png")]:
            if os.path.isfile(cp):
                with Image.open(cp) as cim:
                    cim = cim.convert("RGBA")
                    alpha = np.array(cim)[..., 3]
                    return (alpha > 127)
        m_dir = os.path.join(class_dir, masks_dirname)
        for mp in [os.path.join(m_dir, stem + "_mask.png"), os.path.join(m_dir, stem + ".png")]:
            if os.path.isfile(mp):
                with Image.open(mp) as mim:
                    m = np.array(mim.convert("L"))
                    return (m > 127)
    except Exception:
        return None
    return None

def apply_mask_np(img_rgb: np.ndarray, mask: np.ndarray, gray_bg: Tuple[int, int, int] = (127, 127, 127)) -> np.ndarray:
    """
    Применяет маску к изображению, заменяя фон на серый.
    
    Args:
        img_rgb: RGB изображение [H, W, 3]
        mask: Булева маска [H, W]
        gray_bg: Цвет фона (R, G, B)
        
    Returns:
        Маскированное изображение [H, W, 3]
    """
    bg = np.zeros_like(img_rgb)
    bg[..., 0] = gray_bg[0]
    bg[..., 1] = gray_bg[1]
    bg[..., 2] = gray_bg[2]
    m = mask.astype(bool)
    out = bg.copy()
    out[m] = img_rgb[m]
    return out

def discover_items_only_cropped(data_root: str, log=None) -> Tuple[List[Tuple[str, str]], Dict[str, int], Dict[str, int]]:
    """
    Обнаруживает классы и изображения только из подпапок "cropped".
    
    Сканирует структуру dataset/class_name/cropped/*.jpg и собирает
    метаданные для обучения. Исключает классы с определёнными ключевыми словами.
    Применяет объединение классов согласно MERGE_MAP.
    
    Args:
        data_root: Корневая папка датасета
        log: Опциональный logger
        
    Returns:
        Кортеж из:
        - список пар (путь_к_изображению, имя_класса)
        - словарь {класс: id}
        - словарь {класс: количество_кропов}
    """
    assert os.path.isdir(data_root), f"Папка не найдена: {data_root}"
    SKIP_SUBSTR = [normalize_str("групповые"), normalize_str("тренировк"), normalize_str("линейк")]

    items: List[Tuple[str, str]] = []
    classes_set: set[str] = set()
    counts: Dict[str, int] = {}
    for class_name in sorted(os.listdir(data_root)):
        if class_name.startswith("."):
            continue
        class_dir = os.path.join(data_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        if any(s in normalize_str(class_name) for s in SKIP_SUBSTR):
            continue

        final_class_name = MERGE_MAP.get(class_name, class_name)

        crops_dir = os.path.join(class_dir, "cropped")
        if not os.path.isdir(crops_dir):
            if log:
                log.error(f"[DATA] Класс '{class_name}': нет папки 'cropped' => пропуск класса")
            continue
        cnt = 0
        for f in sorted(os.listdir(crops_dir)):
            p = os.path.join(crops_dir, f)
            if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                # Исключение по DSCN-диапазонам
                try:
                    dscn_match = re.search(r"dscn(\d+)", f, re.IGNORECASE)
                    if dscn_match:
                        dscn_num = int(dscn_match.group(1))
                        is_excluded = any(min_val <= dscn_num <= max_val for min_val, max_val in EXCLUDE_DSCN_RANGES)
                        if is_excluded:
                            if log:
                                log.info(f"[DATA] Исключен файл по DSCN-диапазону: {f}")
                            continue
                except Exception:
                    pass
                items.append((p, final_class_name))
                cnt += 1
        if cnt > 0:
            classes_set.add(final_class_name)
            counts[final_class_name] = counts.get(final_class_name, 0) + cnt
            
    classes = sorted(list(classes_set))
    if log:
        log.info("[DATA] Количество кропов по классам (с учетом объединения):")
        for c in sorted(counts.keys()):
            log.info(f"  - {c}: {counts[c]}")

    assert len(classes) > 0, "Не найдено ни одного класса с изображениями в 'cropped'"
    cls2id = {c: i for i, c in enumerate(classes)}
    return items, cls2id, counts


class AlbWrap:
    """
    Обёртка для аугментаций Albumentations над PIL изображениями.
    
    Преобразует PIL → numpy → Albumentations → результат.
    """
    def __init__(self, aug):
        """
        Args:
            aug: Объект аугментации Albumentations
        """
        self.aug = aug
    def __call__(self, pil_img: Image.Image):
        """
        Применяет аугментацию к PIL изображению.
        
        Args:
            pil_img: Исходное PIL изображение
            
        Returns:
            Обработанное изображение в формате numpy
        """
        arr = np.array(pil_img.convert("RGB"))
        return self.aug(image=arr)["image"]


class AlbDS(Dataset):
    """
    Dataset для обучения эмбеддера с поддержкой масок.
    
    Поддерживает три режима обработки:
    - bbox: Использует изображение как есть
    - mask: Применяет маску (фон заменяется на серый)
    - mix: Смешанный режим (с вероятностью mix_mask_p)
    """
    def __init__(self, items: List[Tuple[str, str]], cls2id: Dict[str, int], aug, crop_mode: str = "bbox", mix_mask_p: float = 0.5,
                 masks_dirname: str = "masks", cutouts_dirname: str = "cutouts", gray_bg: Tuple[int, int, int] = (127, 127, 127)):
        """
        Args:
            items: Список пар (путь_к_изображению, класс)
            cls2id: Маппинг класс → ID
            aug: Объект аугментации Albumentations
            crop_mode: Режим обработки ("bbox", "mask", "mix")
            mix_mask_p: Вероятность использования маски в mix режиме
            masks_dirname: Название папки с масками
            cutouts_dirname: Название папки с RGBA cutouts
            gray_bg: Цвет фона для маскированных изображений
        """
        self.items = items
        self.cls2id = cls2id
        self.aug = aug
        self.crop_mode = crop_mode
        self.mix_mask_p = float(mix_mask_p)
        self.masks_dirname = masks_dirname
        self.cutouts_dirname = cutouts_dirname
        self.gray_bg = gray_bg

    def _find_mask_for(self, img_path: str) -> np.ndarray | None:
        """Пытается найти бинарную маску для данного изображения.
        Поиск: сначала cutouts (RGBA), затем masks (PNG с _mask суффиксом).
        Возвращает маску HxW bool или None.
        """
        class_dir = os.path.dirname(os.path.dirname(img_path))  # .../class/cropped/file -> .../class
        stem = os.path.splitext(os.path.basename(img_path))[0]
        # 1) cutouts: <stem>_cutout.png с альфой
        # Правильный путь cutouts: .../class/cutouts
        cut_dir = os.path.join(class_dir, self.cutouts_dirname)
        cut_path_candidates = [
            os.path.join(cut_dir, stem + "_cutout.png"),
            os.path.join(cut_dir, stem + ".png"),
        ]
        for cp in cut_path_candidates:
            if os.path.isfile(cp):
                try:
                    with Image.open(cp) as cim:
                        cim = cim.convert("RGBA")
                        alpha = np.array(cim)[..., 3]
                        return (alpha > 127)
                except Exception:
                    pass
        # 2) masks: <stem>_mask.png или <stem>.png
        # Правильный путь masks: .../class/masks
        m_dir = os.path.join(class_dir, self.masks_dirname)
        mask_candidates = [
            os.path.join(m_dir, stem + "_mask.png"),
            os.path.join(m_dir, stem + ".png"),
        ]
        for mp in mask_candidates:
            if os.path.isfile(mp):
                try:
                    with Image.open(mp) as mim:
                        m = np.array(mim.convert("L"))
                        return (m > 127)
                except Exception:
                    pass
        return None

    def _apply_mask(self, img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        bg = np.zeros_like(img_rgb)
        bg[..., 0] = self.gray_bg[0]
        bg[..., 1] = self.gray_bg[1]
        bg[..., 2] = self.gray_bg[2]
        m = mask.astype(bool)
        out = bg.copy()
        out[m] = img_rgb[m]
        return out
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i: int):
        p, c = self.items[i]
        img = np.array(Image.open(p).convert("RGB"))
        use_mask = False
        if self.crop_mode == "mask":
            use_mask = True
        elif self.crop_mode == "mix":
            use_mask = (random.random() < self.mix_mask_p)
        if use_mask:
            m = self._find_mask_for(p)
            if m is not None and m.shape[:2] == img.shape[:2]:
                img = self._apply_mask(img, m)
        out = self.aug(image=img)
        x = out["image"]
        y = self.cls2id[c]
        return x, y


class BalancedPerClassSampler(torch.utils.data.Sampler[int]):
    """
    Класс-сбалансированный самплер с ротацией по эпохам.
    
    На каждой эпохе выдаёт ровно n_per_class самплов на каждый класс.
    По мере прохождения эпох окно выборки сдвигается, обеспечивая 
    ротацию используемых самплов.
    
    Поведение:
    - Для каждого класса поддерживает перемешанный список индексов
    - На каждой эпохе выбирает окно длиной n_per_class с сдвигом
    - Обеспечивает сбалансированность классов в каждой эпохе
    """
    def __init__(self, items: List[Tuple[str, str]], cls2id: Dict[str, int], n_per_class: int, seed: int = 42):
        """
        Args:
            items: Список пар (путь_к_изображению, класс)
            cls2id: Маппинг класс -> ID
            n_per_class: Количество самплов на класс на эпоху
            seed: Сид для воспроизводимости
        """
        self.items = items
        self.cls2id = cls2id
        self.n_per_class = int(n_per_class)
        self.seed = int(seed)
        # Построить индексы по классам
        self.class_to_indices: Dict[int, List[int]] = {cid: [] for cid in cls2id.values()}
        for idx, (_, cname) in enumerate(items):
            cid = cls2id[cname]
            self.class_to_indices[cid].append(idx)
        # Перемешанные порядки для стабильной ротации
        self.shuffled_per_class: Dict[int, List[int]] = {}
        rng = random.Random(self.seed)
        for cid, idxs in self.class_to_indices.items():
            order = list(idxs)
            rng.shuffle(order)
            self.shuffled_per_class[cid] = order
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(max(0, epoch))

    def __iter__(self):
        # Для каждого класса выбираем окно длиной n_per_class со сдвигом epoch * n_per_class
        selected: List[int] = []
        for cid, order in self.shuffled_per_class.items():
            L = len(order)
            if L == 0:
                continue
            n = min(self.n_per_class, L)
            offset = (self._epoch * self.n_per_class) % L
            if offset + n <= L:
                take = order[offset:offset + n]
            else:
                take = order[offset:] + order[:(offset + n) % L]
            selected.extend(take)
        # Перемешаем итоговый список, чтобы классы чередовались
        rng = random.Random(self.seed + self._epoch)
        rng.shuffle(selected)
        return iter(selected)

    def __len__(self) -> int:
        # Общее количество элементов на эпоху: n_per_class * num_classes_с_данными
        num_classes_non_empty = sum(1 for idxs in self.class_to_indices.values() if len(idxs) > 0)
        return self.n_per_class * num_classes_non_empty


def count_masks_per_class(data_root: str, masks_dirname: str = "masks", log=None) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for class_name in sorted(os.listdir(data_root)):
        class_dir = os.path.join(data_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        mdir = os.path.join(class_dir, masks_dirname)
        cnt = 0
        if os.path.isdir(mdir):
            for f in os.listdir(mdir):
                p = os.path.join(mdir, f)
                if p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                    cnt += 1
        counts[class_name] = cnt
    if log:
        log.info("[MASKS] Количество масок по классам:")
        for c in sorted(counts.keys()):
            log.info(f"  - {c}: {counts[c]}")
    return counts


def _sam2_build(cfg: str, ckpt: str, device: str):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model = build_sam2(cfg, ckpt).to(device)
    predictor = SAM2ImagePredictor(model)
    return predictor


def generate_masks_for_crops(data_root: str, sam2_cfg: str, sam2_ckpt: str, device: str, log=None,
                             crops_dirname: str = "cropped", masks_dirname: str = "masks") -> None:
    """Генерирует маски для кропов: для каждого изображения из 'cropped' запускает SAM-2
    с bbox равным всему изображению и сохраняет бинарную маску в 'masks'.
    """
    try:
        predictor = _sam2_build(sam2_cfg, sam2_ckpt, device)
    except Exception as e:
        if log:
            log.error(f"[SAM2] Не удалось инициализировать SAM-2: {e}")
        raise
    classes = [d for d in sorted(os.listdir(data_root)) if os.path.isdir(os.path.join(data_root, d))]
    total_imgs = 0
    for c in classes:
        cdir = os.path.join(data_root, c)
        crops_dir = os.path.join(cdir, crops_dirname)
        if not os.path.isdir(crops_dir):
            continue
        for f in os.listdir(crops_dir):
            if str(f).lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                total_imgs += 1
    if log:
        log.info(f"[SAM2] Начинаю генерацию масок для кропов, всего изображений: {total_imgs}")
    pbar = tqdm(total=total_imgs, desc="SAM-2 masks", leave=False)
    for c in classes:
        cdir = os.path.join(data_root, c)
        crops_dir = os.path.join(cdir, crops_dirname)
        if not os.path.isdir(crops_dir):
            continue
        mdir = os.path.join(cdir, masks_dirname)
        os.makedirs(mdir, exist_ok=True)
        for f in sorted(os.listdir(crops_dir)):
            if not str(f).lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                continue
            stem = os.path.splitext(f)[0]
            out_mask = os.path.join(mdir, stem + "_mask.png")
            if os.path.isfile(out_mask):
                pbar.update(1)
                continue
            img_path = os.path.join(crops_dir, f)
            try:
                with Image.open(img_path).convert("RGB") as im:
                    im_np = np.array(im)
                predictor.set_image(im_np)
                H, W = im_np.shape[:2]
                box = np.array([0, 0, W - 1, H - 1], dtype=np.float32)
                masks, ious, _ = predictor.predict(box=box[None, :], multimask_output=True)
                if masks.shape[0] == 0:
                    m = np.zeros((H, W), dtype=np.uint8)
                else:
                    best = int(np.argmax(ious.reshape(-1)))
                    m = (masks[best].astype(np.uint8) * 255)
                Image.fromarray(m).save(out_mask)
            except Exception:
                # Пишем пустую маску как fallback
                try:
                    with Image.open(img_path) as _im:
                        H, W = _im.size[1], _im.size[0]
                except Exception:
                    H, W = 0, 0
                if H > 0 and W > 0:
                    Image.fromarray(np.zeros((H, W), dtype=np.uint8)).save(out_mask)
            pbar.update(1)
    pbar.close()


class EmbedNet(nn.Module):
    """
    Нейросеть для извлечения эмбеддингов на основе DINOv3.
    
    Состоит из предобученного DINOv3 backbone и полносвязной головы.
    Выходные эмбеддинги нормализуются по L2 норме.
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
        # Разрешаем вход произвольного размера
        if hasattr(self.backbone, "patch_embed"):
            try:
                self.backbone.patch_embed.strict_img_size = False
            except Exception:
                pass

        feat_dim = self.backbone.num_features
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)
        z = F.normalize(self.head(f), dim=1)
        return z


class ArcMarginProduct(nn.Module):
    """
    ArcFace margin product для метрического обучения.
    
    Реализует ArcFace loss: добавляет angular margin к косинусному сходству
    между эмбеддингами и весами классов. Это улучшает разделимость классов.
    """
    def __init__(self, in_dim: int, n_classes: int, s: float = 30.0, m: float = 0.30):
        """
        Args:
            in_dim: Размерность входных эмбеддингов
            n_classes: Количество классов
            s: Масштабирующий коэффициент (temperature)
            m: Angular margin в радианах
        """
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, in_dim))
        nn.init.xavier_uniform_(self.W)
        self.s, self.m = s, m
    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        W = F.normalize(self.W, dim=1)
        cos = F.linear(z, W)
        one_hot = F.one_hot(y, num_classes=W.size(0)).float().to(z.device)
        return self.s * (cos - one_hot * self.m)


def train(args):
    """
    Основная функция обучения эмбеддера DINOv3 с ArcFace.
    
    Выполняет полный цикл обучения: подготовка данных → обучение → валидация → 
    экспорт прототипов → тестирование. Поддерживает резюмирование, 
    DataParallel и интеграцию с ClearML.
    
    Args:
        args: Объект argparse с параметрами обучения
    """
    # Создаем папку с номером рана
    run_number = 1
    base_out_dir = args.out_dir
    while os.path.exists(os.path.join(base_out_dir, f"run_{run_number:03d}")):
        run_number += 1
    args.out_dir = os.path.join(base_out_dir, f"run_{run_number:03d}")
    
    log = setup_logging(args.out_dir)
    log.info(f"Запуск #{run_number}, папка: {args.out_dir}")
    set_seed(args.seed)

    # JSONL лог ран-метрик
    metrics_jsonl = os.path.join(args.out_dir, "metrics.jsonl")
    try:
        append_jsonl(metrics_jsonl, {
            "event": "run_start",
            "timestamp": time.time(),
            "run_number": run_number,
            "out_dir": args.out_dir,
            "args": vars(args),
        })
    except Exception:
        pass

    # Инициализация ClearML по флагу
    task = None
    clr_logger = None
    if getattr(args, "use_clearml", False):
        try:
            # ClearML: задайте CLEARML_API_ACCESS_KEY, CLEARML_API_SECRET_KEY, CLEARML_API_HOST в окружении
            from clearml import Task
            task_name = args.clearml_task_name if getattr(args, "clearml_task_name", "") else f"dino3_run_{run_number:03d}"
            task = Task.init(
                project_name=args.clearml_project,
                task_name=task_name,
                auto_connect_frameworks=False,
                auto_connect_arg_parser=False,
            )
            try:
                task.connect(vars(args))
            except Exception:
                pass
            try:
                clr_logger = task.get_logger()
            except Exception:
                clr_logger = None
            log.info("[ClearML] Трекинг включен")
        except Exception as e:
            log.error(f"[ClearML] Не удалось инициализировать: {e}")

    interrupted = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # Конфиги и трансформы от timm для выбранной модели (DINOv3)
    tmp_model = timm.create_model(args.model_id, pretrained=True, num_classes=0, global_pool="token")
    data_cfg = timm.data.resolve_model_data_config(tmp_model)
    if args.img_size is not None:
        data_cfg["input_size"] = (3, args.img_size, args.img_size)

    # Albumentations аугментации
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import inspect

    mean = tuple(data_cfg.get("mean", (0.5, 0.5, 0.5)))
    std = tuple(data_cfg.get("std", (0.5, 0.5, 0.5)))
    pad_value = (114, 114, 114)

    def has_param(transform_cls, param: str) -> bool:
        try:
            return param in inspect.signature(transform_cls.__init__).parameters
        except Exception:
            return False

    # PadIfNeeded kwargs by version
    pad_kwargs = {"min_height": args.img_size, "min_width": args.img_size, "border_mode": 0}
    if has_param(A.PadIfNeeded, "border_value"):
        pad_kwargs["border_value"] = pad_value
    elif has_param(A.PadIfNeeded, "value"):
        pad_kwargs["value"] = pad_value

    # Affine kwargs by version
    affine_kwargs = {
        "rotate": (-180, 180),
        "translate_percent": (-0.02, 0.02),
        "scale": (0.9, 1.1),
        "p": 0.7,
    }
    if has_param(A.Affine, "border_mode"):
        affine_kwargs["border_mode"] = 0
    elif has_param(A.Affine, "mode"):
        affine_kwargs["mode"] = 0
    if has_param(A.Affine, "border_value"):
        affine_kwargs["border_value"] = pad_value
    elif has_param(A.Affine, "cval"):
        affine_kwargs["cval"] = pad_value

    # Noise transform by availability
    if hasattr(A, "GaussianNoise"):
        noise_t = A.GaussianNoise(var_limit=(5.0, 25.0), p=0.2)
    else:
        noise_t = A.GaussNoise(var_limit=(5.0, 25.0), p=0.2)

    # ImageCompression params by version
    if has_param(A.ImageCompression, "quality_range"):
        ic_t = A.ImageCompression(quality_range=(60, 95), p=0.2)
    else:
        ic_t = A.ImageCompression(quality_lower=60, quality_upper=95, p=0.2)

    # Dropout / Cutout by availability
    if hasattr(A, "Cutout"):
        dropout_t = A.Cutout(
            num_holes=4,
            max_h_size=int(0.25 * args.img_size),
            max_w_size=int(0.25 * args.img_size),
            fill_value=pad_value,
            p=0.5,
        )
    else:
        # Fallback to CoarseDropout with version-safe args
        cd_kwargs = {
            "max_holes": 4,
            "max_height": int(0.25 * args.img_size),
            "max_width": int(0.25 * args.img_size),
            "min_holes": 1,
            "min_height": 16,
            "min_width": 16,
            "p": 0.5,
        }
        if has_param(A.CoarseDropout, "fill_value"):
            cd_kwargs["fill_value"] = pad_value
        elif has_param(A.CoarseDropout, "mask_fill_value"):
            cd_kwargs["mask_fill_value"] = pad_value
        dropout_t = A.CoarseDropout(**cd_kwargs)

    train_aug = A.Compose([
        A.LongestMaxSize(max_size=args.img_size),
        A.PadIfNeeded(**pad_kwargs),
        A.Affine(**affine_kwargs),
        A.HorizontalFlip(p=0.5),
        A.Perspective(scale=(0.02, 0.06), p=0.2),
        A.ColorJitter(0.3, 0.3, 0.3, 0.05, p=0.7),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.CLAHE(clip_limit=(1, 3), p=0.1),
        noise_t,
        A.MotionBlur(blur_limit=3, p=0.1),
        ic_t,
        dropout_t,
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    val_aug = A.Compose([
        A.LongestMaxSize(max_size=args.img_size),
        A.PadIfNeeded(**pad_kwargs),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    # Данные (только 'cropped') + валидация достаточности
    items, cls2id, counts = discover_items_only_cropped(args.data_root, log)
    # Проверка минимального количества на класс
    min_required = int(max(1, args.min_per_class))
    lacking = [c for c, k in counts.items() if k < min_required]
    if len(lacking) > 0:
        log.error("[DATA] Недостаточно кропов в классах (min_per_class=%d):" % min_required)
        for c in lacking:
            log.error(f"  - {c}: {counts[c]} < {min_required}")
        raise SystemExit(2)

    # Если обучение по маскам или микс — проверяем наличие масок и по флагу генерируем
    if args.crop_mode in ("mask", "mix"):
        mask_counts = count_masks_per_class(args.data_root, masks_dirname=args.masks_dirname, log=log)
        total_masks = sum(mask_counts.values())
        if total_masks == 0:
            if args.auto_generate_masks:
                log.info("[MASKS] Маски отсутствуют. Запускаю генерацию через SAM-2...")
                generate_masks_for_crops(
                    args.data_root,
                    sam2_cfg=args.sam2_cfg,
                    sam2_ckpt=args.sam2_ckpt,
                    device=("cuda" if torch.cuda.is_available() else "cpu"),
                    log=log,
                    crops_dirname="cropped",
                    masks_dirname=args.masks_dirname,
                )
                # Пересчитаем
                mask_counts = count_masks_per_class(args.data_root, masks_dirname=args.masks_dirname, log=log)
                if sum(mask_counts.values()) == 0:
                    log.error("[MASKS] Не удалось создать маски. Останавливаю работу.")
                    raise SystemExit(3)
            else:
                log.error("[MASKS] Маски отсутствуют. Запустите с --auto_generate_masks или подготовьте маски заранее.")
                raise SystemExit(3)

    labels_all = [c for _, c in items]
    id2cls = {v: k for k, v in cls2id.items()}
    from sklearn.model_selection import train_test_split
    # Cначала отделим test, затем из остатка отделим val
    test_size = float(args.test_split)
    val_size = float(args.val_split)
    if test_size > 0:
        items_tmp, test_items = train_test_split(
            items, test_size=test_size, stratify=labels_all, random_state=args.seed
        )
        labels_tmp = [c for _, c in items_tmp]
    else:
        items_tmp, test_items = items, []
        labels_tmp = labels_all
    val_rel = val_size / max(1.0, (1.0 - test_size))
    if val_size > 0:
        train_items, val_items = train_test_split(
            items_tmp, test_size=val_rel, stratify=labels_tmp, random_state=args.seed
        )
    else:
        train_items, val_items = items_tmp, []

    # Сохранение сплитов
    try:
        splits_path = os.path.join(args.out_dir, "splits.json")
        def to_rec(lst: List[Tuple[str, str]]):
            return [{"path": p, "class": c} for (p, c) in lst]
        with open(splits_path, "w", encoding="utf-8") as f:
            json.dump({
                "train": to_rec(train_items),
                "val": to_rec(val_items),
                "test": to_rec(test_items),
            }, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    pin = torch.cuda.is_available()
    # Тренировочный лоадер: опционально класс-балансный самплер на N элементов/класс
    sampler = None
    if args.balanced_per_class:
        sampler = BalancedPerClassSampler(train_items, cls2id, n_per_class=args.samples_per_class, seed=args.seed)
        dl_train = DataLoader(AlbDS(train_items, cls2id, train_aug, crop_mode=args.crop_mode, mix_mask_p=args.mix_mask_p, masks_dirname=args.masks_dirname), batch_size=args.batch, shuffle=False,
                              sampler=sampler, num_workers=args.workers, pin_memory=pin,
                              persistent_workers=pin and args.workers > 0)
    else:
        dl_train = DataLoader(AlbDS(train_items, cls2id, train_aug, crop_mode=args.crop_mode, mix_mask_p=args.mix_mask_p, masks_dirname=args.masks_dirname), batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=pin, persistent_workers=pin and args.workers > 0)
    dl_val = None
    if val_items:
        dl_val = DataLoader(AlbDS(val_items, cls2id, val_aug, crop_mode="bbox"), batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=pin, persistent_workers=pin and args.workers > 0)
    dl_test = None
    if test_items:
        dl_test = DataLoader(AlbDS(test_items, cls2id, val_aug, crop_mode="bbox"), batch_size=args.batch, shuffle=False,
                             num_workers=args.workers, pin_memory=pin, persistent_workers=pin and args.workers > 0)

    # Модель и критерии
    model = EmbedNet(dim=args.emb_dim, model_name=args.model_id, pool="token",
                     freeze_backbone=args.freeze_backbone, img_size=args.img_size).to(device)
    arc = ArcMarginProduct(args.emb_dim, n_classes=len(cls2id)).to(device)
    ce = nn.CrossEntropyLoss()

    # DataParallel при наличии нескольких GPU по флагу
    use_dp = (args.gpus is not None and args.gpus > 1 and torch.cuda.is_available() and torch.cuda.device_count() >= args.gpus)
    if use_dp:
        model = nn.DataParallel(model, device_ids=list(range(args.gpus)))
        arc = nn.DataParallel(arc, device_ids=list(range(args.gpus)))

    # Оптимизатор
    if args.freeze_backbone:
        params = list(model.head.parameters()) + list(arc.parameters())
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    else:
        opt = torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": max(args.lr * 0.2, 5e-5), "weight_decay": 0.05},
            {"params": model.head.parameters(),     "lr": args.lr,             "weight_decay": 1e-4},
            {"params": arc.parameters(),            "lr": args.lr,             "weight_decay": 1e-4},
        ])

    scaler = torch.amp.GradScaler(device="cuda", enabled=torch.cuda.is_available())
    autocast_ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available())
                    if torch.cuda.is_available() else torch.autocast(device_type="cpu", enabled=False))

    # Сохранение конфигурации (включая оптимизатор и лосс)
    os.makedirs(args.out_dir, exist_ok=True)
    config_out = dict(vars(args))
    config_out.update({
        "optimizer": {
            "name": opt.__class__.__name__,
            "param_groups": len(opt.param_groups),
            "lr": args.lr,
        },
        "loss": {
            "name": "CrossEntropyLoss + ArcFace margin",
            "arcface_s": getattr(arc.module if isinstance(arc, nn.DataParallel) else arc, 's', 30.0) if hasattr(ArcMarginProduct, '__init__') else 30.0,
            "arcface_m": getattr(arc.module if isinstance(arc, nn.DataParallel) else arc, 'm', 0.30) if hasattr(ArcMarginProduct, '__init__') else 0.30,
        },
        "data_parallel": bool(use_dp),
        "num_devices": (args.gpus if use_dp else 1),
        "num_classes": len(cls2id),
    })
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_out, f, ensure_ascii=False, indent=2)

    best_acc = 0.0
    start_epoch = 1
    last_ckpt_path = os.path.join(args.out_dir, "checkpoint_last.pth")
    best_ckpt_path = os.path.join(args.out_dir, "checkpoint_best.pth")

    # Резюмирование
    resume_path = args.resume if args.resume else (last_ckpt_path if os.path.exists(last_ckpt_path) else None)
    if resume_path and os.path.exists(resume_path):
        try:
            ckpt = torch.load(resume_path, map_location=device)
            model.load_state_dict(ckpt["model"])  # type: ignore[index]
            arc.load_state_dict(ckpt["arc"])      # type: ignore[index]
            opt.load_state_dict(ckpt["opt"])      # type: ignore[index]
            best_acc = float(ckpt.get("best_acc", 0.0))
            start_epoch = int(ckpt.get("epoch", 1)) + 1
            log.info(f"[RESUME] Продолжаем с эпохи {start_epoch} (best_acc={best_acc:.3f}) из {resume_path}")
        except Exception as e:
            log.warning(f"[RESUME] Не удалось загрузить чекпоинт: {e}")

    def _module_or_self(m: nn.Module) -> nn.Module:
        return m.module if isinstance(m, nn.DataParallel) else m

    def save_ckpt(path: str, epoch: int, best_acc_val: float):
        tmp = path + ".tmp"
        torch.save({
            "model": _module_or_self(model).state_dict(),
            "arc": _module_or_self(arc).state_dict(),
            "opt": opt.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc_val,
            "cls2id": cls2id,
            "model_name": args.model_id,
            "emb_dim": args.emb_dim,
            "freeze_backbone": args.freeze_backbone,
            "img_size": args.img_size,
            "data_cfg": data_cfg,
            "data_parallel": bool(use_dp),
        }, tmp)
        os.replace(tmp, path)

    # Обучение
    try:
        for ep in range(start_epoch, args.epochs + 1):
            model.train(); arc.train()
            # Ротация самплера по эпохам
            if sampler is not None and hasattr(sampler, "set_epoch"):
                try:
                    sampler.set_epoch(ep - 1)
                except Exception:
                    pass
            tot = 0.0; cnt = 0
            train_correct = 0; train_total = 0
            y_true_tr: List[int] = []
            y_prob_tr: List[np.ndarray] = []
            pbar = tqdm(dl_train, desc=f"Epoch {ep}/{args.epochs}", leave=False)
            for x, y in pbar:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with autocast_ctx:
                    z = model(x)
                    logits = arc(z, y)
                    loss = ce(logits, y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                tot += loss.item() * x.size(0); cnt += x.size(0)
                # train running acc and probs for metrics
                with torch.no_grad():
                    pred = logits.argmax(1)
                    train_correct += (pred == y).sum().item(); train_total += y.size(0)
                    probs = torch.softmax(logits.detach().float(), dim=1).cpu().numpy()
                    y_prob_tr.append(probs)
                    y_true_tr.extend(y.detach().cpu().tolist())
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc_tr": f"{train_correct/max(1,train_total):.3f}"})

            # Валидация
            model.eval(); arc.eval()
            val_acc = float('nan'); f1_val = float('nan'); auc_val = float('nan')
            if dl_val is not None:
                correct = 0; total = 0
                y_true_val: List[int] = []
                y_prob_val: List[np.ndarray] = []
                with torch.no_grad():
                    for x, y in dl_val:
                        x = x.to(device, non_blocking=True)
                        y = y.to(device, non_blocking=True)
                        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                            z = model(x)
                            logits = arc(z, y)
                        pred = logits.argmax(1)
                        correct += (pred == y).sum().item(); total += y.size(0)
                        probs = torch.softmax(logits.detach().float(), dim=1).cpu().numpy()
                        y_prob_val.append(probs)
                        y_true_val.extend(y.detach().cpu().tolist())
                val_acc = correct / max(1, total)
            # Метрики train
            train_acc = train_correct / max(1, train_total)
            try:
                y_prob_tr_arr = np.concatenate(y_prob_tr, axis=0) if y_prob_tr else np.zeros((0, len(cls2id)))
                y_true_tr_arr = np.array(y_true_tr, dtype=int)
                y_pred_tr_arr = np.argmax(y_prob_tr_arr, axis=1) if y_prob_tr_arr.size else np.array([], dtype=int)
                f1_tr = f1_score(y_true_tr_arr, y_pred_tr_arr, average='macro') if y_prob_tr_arr.size else float('nan')
                auc_tr = roc_auc_score(y_true_tr_arr, y_prob_tr_arr, multi_class='ovr', average='macro') if y_prob_tr_arr.size else float('nan')
                # Per-class metrics (train)
                per_prec_tr, per_rec_tr, per_f1_tr, per_sup_tr = precision_recall_fscore_support(
                    y_true_tr_arr, y_pred_tr_arr, labels=list(range(len(cls2id))), average=None, zero_division=0
                ) if y_prob_tr_arr.size else (np.array([]), np.array([]), np.array([]), np.array([]))
                # Per-class AUC (one-vs-rest)
                per_auc_tr_list: List[float] = []
                if y_prob_tr_arr.size:
                    for k in range(len(cls2id)):
                        try:
                            y_true_bin = (y_true_tr_arr == k).astype(int)
                            if y_true_bin.max() == y_true_bin.min():
                                raise ValueError("single-class")
                            per_auc_tr_list.append(float(roc_auc_score(y_true_bin, y_prob_tr_arr[:, k])))
                        except Exception:
                            per_auc_tr_list.append(float('nan'))
                per_auc_tr = np.array(per_auc_tr_list, dtype=float) if y_prob_tr_arr.size else np.array([])
            except Exception:
                f1_tr, auc_tr = float('nan'), float('nan')
                per_prec_tr = per_rec_tr = per_f1_tr = per_sup_tr = per_auc_tr = np.array([])
            # Метрики val
            try:
                y_prob_val_arr = np.concatenate(y_prob_val, axis=0) if y_prob_val else np.zeros((0, len(cls2id)))
                y_true_val_arr = np.array(y_true_val, dtype=int)
                y_pred_val_arr = np.argmax(y_prob_val_arr, axis=1) if y_prob_val_arr.size else np.array([], dtype=int)
                f1_val = f1_score(y_true_val_arr, y_pred_val_arr, average='macro') if y_prob_val_arr.size else float('nan')
                auc_val = roc_auc_score(y_true_val_arr, y_prob_val_arr, multi_class='ovr', average='macro') if y_prob_val_arr.size else float('nan')
                # Per-class metrics (val)
                per_prec_val, per_rec_val, per_f1_val, per_sup_val = precision_recall_fscore_support(
                    y_true_val_arr, y_pred_val_arr, labels=list(range(len(cls2id))), average=None, zero_division=0
                ) if y_prob_val_arr.size else (np.array([]), np.array([]), np.array([]), np.array([]))
                # Per-class AUC (one-vs-rest)
                per_auc_val_list: List[float] = []
                if y_prob_val_arr.size:
                    for k in range(len(cls2id)):
                        try:
                            y_true_bin = (y_true_val_arr == k).astype(int)
                            if y_true_bin.max() == y_true_bin.min():
                                raise ValueError("single-class")
                            per_auc_val_list.append(float(roc_auc_score(y_true_bin, y_prob_val_arr[:, k])))
                        except Exception:
                            per_auc_val_list.append(float('nan'))
                per_auc_val = np.array(per_auc_val_list, dtype=float) if y_prob_val_arr.size else np.array([])
            except Exception:
                f1_val, auc_val = float('nan'), float('nan')
                per_prec_val = per_rec_val = per_f1_val = per_sup_val = per_auc_val = np.array([])

            log.info(
                f"Epoch {ep:02d} | train_loss={tot/max(1,cnt):.4f} | "
                f"acc_tr={train_acc:.3f} f1_tr={f1_tr:.3f} auc_tr={auc_tr:.3f} | "
                f"acc_val={val_acc:.3f} f1_val={f1_val:.3f} auc_val={auc_val:.3f}"
            )

            # JSONL + ClearML лог метрик эпохи
            try:
                epoch_record = {
                    "event": "epoch_end",
                    "epoch": ep,
                    "timestamp": time.time(),
                    "train": {
                        "loss": float(tot/max(1,cnt)),
                        "acc": float(train_acc),
                        "f1_macro": float(f1_tr),
                        "auc_macro": float(auc_tr),
                    },
                    "val": {
                        "acc": float(val_acc),
                        "f1_macro": float(f1_val),
                        "auc_macro": float(auc_val),
                    },
                }
                # Per-class into JSONL (optional, for analysis)
                if per_f1_tr.size:
                    epoch_record["train_per_class"] = {
                        id2cls[i]: {
                            "precision": float(per_prec_tr[i]),
                            "recall": float(per_rec_tr[i]),
                            "f1": float(per_f1_tr[i]),
                            "auc": (float(per_auc_tr[i]) if i < per_auc_tr.size and not np.isnan(per_auc_tr[i]) else None),
                            "support": int(per_sup_tr[i]),
                        } for i in range(len(per_f1_tr))
                    }
                if 'per_f1_val' in locals() and np.size(per_f1_val):
                    epoch_record["val_per_class"] = {
                        id2cls[i]: {
                            "precision": float(per_prec_val[i]),
                            "recall": float(per_rec_val[i]),
                            "f1": float(per_f1_val[i]),
                            "auc": (float(per_auc_val[i]) if i < per_auc_val.size and not np.isnan(per_auc_val[i]) else None),
                            "support": int(per_sup_val[i]),
                        } for i in range(len(per_f1_val))
                    }
                append_jsonl(metrics_jsonl, epoch_record)
            except Exception:
                pass
            if clr_logger is not None:
                try:
                    clr_logger.report_scalar("train", "loss", float(tot/max(1,cnt)), iteration=ep)
                    clr_logger.report_scalar("train", "acc", float(train_acc), iteration=ep)
                    clr_logger.report_scalar("train", "f1_macro", float(f1_tr), iteration=ep)
                    clr_logger.report_scalar("train", "auc_macro", float(auc_tr), iteration=ep)
                    clr_logger.report_scalar("val", "acc", float(val_acc), iteration=ep)
                    clr_logger.report_scalar("val", "f1_macro", float(f1_val), iteration=ep)
                    clr_logger.report_scalar("val", "auc_macro", float(auc_val), iteration=ep)
                    # Per-class scalars
                    if per_f1_tr.size:
                        for i in range(len(per_f1_tr)):
                            cname = id2cls.get(i, str(i))
                            clr_logger.report_scalar("train/per_class/precision", cname, float(per_prec_tr[i]), iteration=ep)
                            clr_logger.report_scalar("train/per_class/recall", cname, float(per_rec_tr[i]), iteration=ep)
                            clr_logger.report_scalar("train/per_class/f1", cname, float(per_f1_tr[i]), iteration=ep)
                            if i < per_auc_tr.size and not np.isnan(per_auc_tr[i]):
                                clr_logger.report_scalar("train/per_class/auc", cname, float(per_auc_tr[i]), iteration=ep)
                    if 'per_f1_val' in locals() and np.size(per_f1_val):
                        for i in range(len(per_f1_val)):
                            cname = id2cls.get(i, str(i))
                            clr_logger.report_scalar("val/per_class/precision", cname, float(per_prec_val[i]), iteration=ep)
                            clr_logger.report_scalar("val/per_class/recall", cname, float(per_rec_val[i]), iteration=ep)
                            clr_logger.report_scalar("val/per_class/f1", cname, float(per_f1_val[i]), iteration=ep)
                            if i < per_auc_val.size and not np.isnan(per_auc_val[i]):
                                clr_logger.report_scalar("val/per_class/auc", cname, float(per_auc_val[i]), iteration=ep)
                except Exception:
                    pass

            # Сохранение чекпоинтов
            save_ckpt(last_ckpt_path, ep, best_acc)
            if (ep % max(1, args.save_every)) == 0:
                save_ckpt(os.path.join(args.out_dir, f"checkpoint_ep{ep:03d}.pth"), ep, best_acc)
            if val_acc > best_acc:
                best_acc = val_acc
                save_ckpt(best_ckpt_path, ep, best_acc)
                log.info(f"🌟 Новый лучший чекпоинт сохранён (acc={best_acc:.3f})")
                try:
                    append_jsonl(metrics_jsonl, {
                        "event": "best_update",
                        "epoch": ep,
                        "timestamp": time.time(),
                        "best_acc": float(best_acc),
                    })
                except Exception:
                    pass
                if clr_logger is not None:
                    try:
                        clr_logger.report_scalar("val", "best_acc", float(best_acc), iteration=ep)
                    except Exception:
                        pass

    except KeyboardInterrupt:
        log.warning("⛔ Остановка по KeyboardInterrupt — сохраняю interrupt-чекпоинт...")
        save_ckpt(os.path.join(args.out_dir, "checkpoint_interrupt.pth"), ep if 'ep' in locals() else 0, best_acc)
        interrupted = True
        try:
            append_jsonl(metrics_jsonl, {
                "event": "interrupted",
                "epoch": (ep if 'ep' in locals() else 0),
                "timestamp": time.time(),
            })
        except Exception:
            pass
        if clr_logger is not None:
            try:
                clr_logger.report_text("Training interrupted by KeyboardInterrupt")
            except Exception:
                pass
    except Exception as e:
        log.exception(f"💥 Аварийная ошибка обучения: {e}")
        raise

    # Если прервали вручную — корректно завершим без пост-этапов
    if interrupted:
        log.info("✅ Обучение завершено корректно (KeyboardInterrupt).")
        try:
            append_jsonl(metrics_jsonl, {"event": "run_end", "status": "interrupted", "timestamp": time.time()})
        except Exception:
            pass
        if task is not None:
            try:
                task.close()
            except Exception:
                pass
        return

    # Экспорт прототипов из всех доступных кропов (centroids и stacks)
    log.info("[PROTO] Считаю прототипы (centroids/stacks)...")
    # Предподсчёт общего числа изображений для tqdm
    try:
        total_proto = 0
        for c in sorted(cls2id.keys()):
            class_dir = os.path.join(args.data_root, c)
            search_dir = os.path.join(class_dir, "cropped")
            if not os.path.isdir(search_dir):
                continue
            img_paths_tmp = []
            for f in sorted(os.listdir(search_dir)):
                p = os.path.join(search_dir, f)
                if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    img_paths_tmp.append(p)
            if args.max_refs_per_class is not None and args.max_refs_per_class > 0:
                img_paths_tmp = img_paths_tmp[:args.max_refs_per_class]
            total_proto += len(img_paths_tmp)
    except Exception:
        total_proto = None  # fallback на неизвестный total

    tfm_proto = AlbWrap(val_aug)

    # Модель на CPU для инференса прототипов (экономия VRAM)
    proto_model = EmbedNet(args.emb_dim, model_name=args.model_id,
                           freeze_backbone=True, img_size=args.img_size)
    proto_model.load_state_dict(model.state_dict(), strict=False)
    proto_model.eval()

    centroids: Dict[str, np.ndarray] = {}
    stacks: Dict[str, np.ndarray] = {}
    with torch.no_grad():
        pbar_proto = tqdm(total=(total_proto or 0), desc="Prototypes", leave=False) if total_proto else None
        # JSON событие старта
        try:
            append_jsonl(metrics_jsonl, {"event": "proto_start", "timestamp": time.time(), "total_images": int(total_proto or 0)})
        except Exception:
            pass
        processed_proto = 0
        for c in sorted(cls2id.keys()):
            # берем все изображения класса из dataset ТОЛЬКО из 'cropped'
            class_dir = os.path.join(args.data_root, c)
            search_dir = os.path.join(class_dir, "cropped")
            if not os.path.isdir(search_dir):
                continue
            # Собираем список путей-эталонов
            img_paths = []
            for f in sorted(os.listdir(search_dir)):
                p = os.path.join(search_dir, f)
                if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    img_paths.append(p)

            # Применяем лимит, если задан
            if args.max_refs_per_class is not None and args.max_refs_per_class > 0:
                img_paths = img_paths[:args.max_refs_per_class]

            vecs: List[np.ndarray] = []
            for p in img_paths:
                try:
                    with Image.open(p).convert("RGB") as im:
                        x = tfm_proto(im).unsqueeze(0)  # CPU tensor
                    z = proto_model(x).squeeze(0).detach().cpu().numpy()
                    z /= max(np.linalg.norm(z), 1e-12)
                    vecs.append(z)
                except Exception:
                    continue
                processed_proto += 1
                if pbar_proto is not None:
                    pbar_proto.update(1)
            if vecs:
                M = np.stack(vecs, axis=0).astype(np.float32)
                mu = M.mean(0); mu /= max(np.linalg.norm(mu), 1e-12)
                centroids[c] = mu
                stacks[c] = M
        if pbar_proto is not None:
            pbar_proto.close()
        # JSON событие завершения
        try:
            total_centroids = len(centroids)
            total_refs = int(sum(stacks[k].shape[0] for k in stacks)) if stacks else 0
            append_jsonl(metrics_jsonl, {"event": "proto_end", "timestamp": time.time(), "processed_images": int(processed_proto), "classes": int(total_centroids), "total_refs": total_refs})
        except Exception:
            pass

    proto_path = os.path.join(args.out_dir, "prototypes.npz")
    np.savez_compressed(proto_path,
                        centroids=centroids, stacks=stacks,
                        cls2id=cls2id, id2cls={v: k for k, v in cls2id.items()})
    log.info(f"✅ Прототипы сохранены: {proto_path}")
    log.info(f"✅ Лучший чекпоинт: {best_ckpt_path if os.path.exists(best_ckpt_path) else last_ckpt_path}")

    # Итоговая оценка на тесте (если есть)
    if dl_test is not None:
        log.info("[TEST] Начинаю оценку на тесте...")
        correct = 0; total = 0
        y_true_te: List[int] = []
        y_prob_te: List[np.ndarray] = []
        with torch.no_grad():
            pbar_test = tqdm(dl_test, desc="Test eval", leave=False)
            for x, y in pbar_test:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    z = model(x)
                    logits = arc(z, y)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item(); total += y.size(0)
                probs = torch.softmax(logits.detach().float(), dim=1).cpu().numpy()
                y_prob_te.append(probs)
                y_true_te.extend(y.detach().cpu().tolist())
                # Обновим прогресс
                if total > 0:
                    pbar_test.set_postfix({"acc": f"{correct/max(1,total):.3f}"})
            pbar_test.close()
        test_acc = correct / max(1, total)
        try:
            y_prob_te_arr = np.concatenate(y_prob_te, axis=0) if y_prob_te else np.zeros((0, len(cls2id)))
            y_true_te_arr = np.array(y_true_te, dtype=int)
            y_pred_te_arr = np.argmax(y_prob_te_arr, axis=1) if y_prob_te_arr.size else np.array([], dtype=int)
            f1_te = f1_score(y_true_te_arr, y_pred_te_arr, average='macro') if y_prob_te_arr.size else float('nan')
            auc_te = roc_auc_score(y_true_te_arr, y_prob_te_arr, multi_class='ovr', average='macro') if y_prob_te_arr.size else float('nan')
            # Per-class test metrics
            per_prec_te, per_rec_te, per_f1_te, per_sup_te = precision_recall_fscore_support(
                y_true_te_arr, y_pred_te_arr, labels=list(range(len(cls2id))), average=None, zero_division=0
            ) if y_prob_te_arr.size else (np.array([]), np.array([]), np.array([]), np.array([]))
            per_auc_te_list: List[float] = []
            if y_prob_te_arr.size:
                for k in range(len(cls2id)):
                    try:
                        y_true_bin = (y_true_te_arr == k).astype(int)
                        if y_true_bin.max() == y_true_bin.min():
                            raise ValueError("single-class")
                        per_auc_te_list.append(float(roc_auc_score(y_true_bin, y_prob_te_arr[:, k])))
                    except Exception:
                        per_auc_te_list.append(float('nan'))
            per_auc_te = np.array(per_auc_te_list, dtype=float) if y_prob_te_arr.size else np.array([])
        except Exception:
            f1_te, auc_te = float('nan'), float('nan')
            per_prec_te = per_rec_te = per_f1_te = per_sup_te = per_auc_te = np.array([])
        log.info(f"[TEST] acc={test_acc:.3f} f1={f1_te:.3f} auc={auc_te:.3f}")

        # Визуализации "до/после" на тесте (bbox vs mask) — сохраняем первые N примеров
        try:
            vis_dir = os.path.join(args.out_dir, "vis_test")
            os.makedirs(vis_dir, exist_ok=True)
            N = 64
            from torchvision.utils import save_image as tv_save_image  # lazy import
            count_saved = 0
            # Список тестовых элементов
            test_list = test_items[:N]
            for (p, c) in test_list:
                try:
                    with Image.open(p).convert("RGB") as im:
                        img_np = np.array(im)
                    # 1) bbox как есть
                    x_bbox = val_aug(image=img_np)["image"].unsqueeze(0)
                    z_bbox = _module_or_self(model)(x_bbox.to(device)).squeeze(0).detach().cpu().numpy()
                    z_bbox /= max(np.linalg.norm(z_bbox), 1e-12)
                    # 2) mask если найдётся
                    mask = find_mask_for(p, masks_dirname=args.masks_dirname)
                    z_mask = None
                    if mask is not None and mask.shape[:2] == img_np.shape[:2]:
                        masked = apply_mask_np(img_np, mask)
                        x_mask = val_aug(image=masked)["image"].unsqueeze(0)
                        z_mask = _module_or_self(model)(x_mask.to(device)).squeeze(0).detach().cpu().numpy()
                        z_mask /= max(np.linalg.norm(z_mask), 1e-12)
                    # Экспорт визуализаций
                    stem = os.path.splitext(os.path.basename(p))[0]
                    # сохраняем исходник и опционально маскированный
                    Image.fromarray(img_np).save(os.path.join(vis_dir, f"{stem}_bbox.jpg"))
                    if mask is not None and z_mask is not None:
                        Image.fromarray(mask.astype(np.uint8) * 255).save(os.path.join(vis_dir, f"{stem}_mask.png"))
                        Image.fromarray(apply_mask_np(img_np, mask)).save(os.path.join(vis_dir, f"{stem}_masked.jpg"))
                    count_saved += 1
                except Exception:
                    continue
            try:
                append_jsonl(metrics_jsonl, {"event": "test_vis", "timestamp": time.time(), "saved": int(count_saved)})
            except Exception:
                pass
        except Exception:
            pass
        try:
            append_jsonl(metrics_jsonl, {
                "event": "test_end",
                "timestamp": time.time(),
                "test": {
                    "acc": float(test_acc),
                    "f1_macro": float(f1_te),
                    "auc_macro": float(auc_te),
                },
                "test_per_class": {
                    id2cls[i]: {
                        "precision": float(per_prec_te[i]),
                        "recall": float(per_rec_te[i]),
                        "f1": float(per_f1_te[i]),
                        "auc": (float(per_auc_te[i]) if i < per_auc_te.size and not np.isnan(per_auc_te[i]) else None),
                        "support": int(per_sup_te[i]),
                    } for i in range(len(per_f1_te))
                } if np.size(per_f1_te) else {},
            })
        except Exception:
            pass
        if clr_logger is not None:
            try:
                clr_logger.report_scalar("test", "acc", float(test_acc), iteration=args.epochs)
                clr_logger.report_scalar("test", "f1_macro", float(f1_te), iteration=args.epochs)
                clr_logger.report_scalar("test", "auc_macro", float(auc_te), iteration=args.epochs)
                if np.size(per_f1_te):
                    for i in range(len(per_f1_te)):
                        cname = id2cls.get(i, str(i))
                        clr_logger.report_scalar("test/per_class/precision", cname, float(per_prec_te[i]), iteration=args.epochs)
                        clr_logger.report_scalar("test/per_class/recall", cname, float(per_rec_te[i]), iteration=args.epochs)
                        clr_logger.report_scalar("test/per_class/f1", cname, float(per_f1_te[i]), iteration=args.epochs)
                        if i < per_auc_te.size and not np.isnan(per_auc_te[i]):
                            clr_logger.report_scalar("test/per_class/auc", cname, float(per_auc_te[i]), iteration=args.epochs)
            except Exception:
                pass

    # Финальный статус рана
    try:
        append_jsonl(metrics_jsonl, {"event": "run_end", "status": "completed", "timestamp": time.time(), "best_acc": float(best_acc)})
    except Exception:
        pass
    if task is not None:
        try:
            task.close()
        except Exception:
            pass
    log.info("✅ Обучение успешно завершено.")


def parse_args():
    """
    Парсинг аргументов командной строки.
    
    Примеры запуска:
    # Базовое обучение на 15 эпох с замороженным backbone
    python lct-dino-3.py --data_root /path/to/dataset --freeze_backbone
    
    # Обучение с разморозкой backbone и балансировкой классов  
    python lct-dino-3.py --data_root /path/to/dataset --epochs 25 --samples_per_class 200 --balanced_per_class
    
    # Режим с масками и автогенерацией через SAM-2
    python lct-dino-3.py --data_root /path/to/dataset --crop_mode mask --auto_generate_masks
    
    # Смешанный режим (bbox + mask) с DataParallel на 2 GPU
    python lct-dino-3.py --data_root /path/to/dataset --crop_mode mix --mix_mask_p 0.3 --gpus 2
    
    # Резюмирование обучения с ClearML трекингом
    python lct-dino-3.py --data_root /path/to/dataset --resume /path/to/checkpoint.pth --use_clearml
    
    # Маленькая модель DINOv3 с кастомными параметрами
    python lct-dino-3.py --model_id "vit_small_patch14_dinov3.lvd142m" --emb_dim 64 --img_size 224 --lr 2e-3
    """
    p = argparse.ArgumentParser(description="Обучение эмбеддера (ArcFace) на DINOv3")
    # Пути
    p.add_argument("--data_root", type=str, default="/home/ubuntu/diabert/dataset/dataset",
                   help="Корневая папка dataset с подпапками-классами и подкаталогами 'cropped'")
    p.add_argument("--out_dir", type=str, default="/home/ubuntu/diabert/dataset/predrazmetka_dashi/lct_dino3_out",
                   help="Папка для логов/чекпоинтов/прототипов")
    # Модель
    p.add_argument("--model_id", type=str, default="vit_large_patch16_dinov3.lvd1689m",
                   help="Идентификатор timm модели (DINOv3)")
    p.add_argument("--emb_dim", type=int, default=128, help="Размер эмбеддинга")
    p.add_argument("--freeze_backbone", action="store_true", help="Заморозить backbone на этапе обучения")
    # Обучение
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--test_split", type=float, default=0.1)
    p.add_argument("--save_every", type=int, default=5,
                   help="Каждые N эпох сохранить именованный чекпоинт")
    p.add_argument("--resume", type=str, default="",
                   help="Путь к чекпоинту для продолжения (по умолчанию ищется checkpoint_last.pth в out_dir)")
    p.add_argument("--max_refs_per_class", type=int, default=None,
                   help="Максимум эталонов на класс при построении прототипов (None — без ограничения)")
    p.add_argument("--gpus", type=int, default=None, help="Число GPU для DataParallel (например, 2)")
    # Данные
    p.add_argument("--min_per_class", type=int, default=300, help="Минимально требуемое число кропов на класс")
    p.add_argument("--masks_dirname", type=str, default="masks", help="Имя папки с масками внутри класса")
    # Класс-балансный самплер
    p.add_argument("--balanced_per_class", action="store_true", help="Включить выборку по N на класс с ротацией")
    p.add_argument("--samples_per_class", type=int, default=300, help="Число элементов на класс в эпоху при балансировке")
    # Режим кропов (bbox/mask/mix) и генерация масок SAM-2
    p.add_argument("--crop_mode", type=str, default="bbox", choices=["bbox", "mask", "mix"],
                   help="Как формировать вход: bbox (как есть), mask (фон зануляем), mix (смешанный режим)")
    p.add_argument("--mix_mask_p", type=float, default=0.4, help="Вероятность выбора маскированного варианта в режиме mix")
    p.add_argument("--auto_generate_masks", action="store_true", help="Автоматически сгенерировать маски через SAM-2, если их нет")
    p.add_argument("--sam2_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml", help="SAM-2 cfg")
    p.add_argument("--sam2_ckpt", type=str, default="/home/ubuntu/sam2/checkpoints/sam2.1_hiera_large.pt", help="SAM-2 веса")
    # Логгирование/трекинг
    p.add_argument("--use_clearml", action="store_true", help="Включить трекинг метрик в ClearML")
    p.add_argument("--clearml_project", type=str, default="DINOv3-Embeddings", help="Имя проекта ClearML")
    p.add_argument("--clearml_task_name", type=str, default="", help="Имя задачи ClearML (по умолчанию генерируется)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)


