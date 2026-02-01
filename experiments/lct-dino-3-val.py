"""
Валидация обученного DINOv3 эмбеддера

Этот скрипт выполняет валидацию и тестирование обученной модели эмбеддера 
на сохранённых сплитах данных. Загружает чекпоинт модели и вычисляет
детальные метрики качества классификации.

Основная логика:
1. Загружает обученную модель из чекпоинта
2. Использует сохранённые сплиты данных (train/val/test) из обучения
3. Выполняет инференс на val и/или test наборах
4. Вычисляет accuracy, F1-score, AUC для каждого класса и макро-усреднение
5. Сохраняет детальные метрики в JSONL формате
6. Опционально создаёт визуализации сравнения bbox vs mask режимов

Требует:
- Чекпоинт обученной модели (.pth файл)
- Файл splits.json с разбиением данных
- Структуру данных как в обучающем скрипте
"""

# -*- coding: utf-8 -*-
import os, sys, json, argparse, time
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


# Опциональная карта объединения классов (как в обучающем скрипте)
MERGE_MAP = {
    "1 Отвертка «-»": "Отвертка",
    "2 Отвертка «+»": "Отвертка",
    "3 Отвертка на смещенный крест": "Отвертка",
}


def setup_logging(out_dir: str):
    """
    Настраивает логгинг для валидации.
    
    Args:
        out_dir: Папка для сохранения логов
        
    Returns:
        Объект logger
    """
    import logging
    os.makedirs(out_dir, exist_ok=True)
    log = logging.getLogger("val")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(out_dir, "val.log"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter(fmt="%Y-%m-%d %H:%M:%S | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch.setFormatter(fmt); fh.setFormatter(fmt)
    log.addHandler(ch); log.addHandler(fh)
    return log


def set_seed(seed: int = 42):
    """
    Устанавливает сид для воспроизводимости.
    
    Args:
        seed: Значение сида
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    """
    Дописывает метрики в JSONL файл.
    
    Args:
        path: Путь к JSONL файлу
        record: Словарь с метриками
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")
    except Exception:
        pass


def discover_items_only_cropped(data_root: str) -> List[Tuple[str, str]]:
    """
    Обнаруживает все изображения из подпапок "cropped".
    
    Args:
        data_root: Корневая папка датасета
        
    Returns:
        Список пар (путь_к_изображению, класс)
    """
    items: List[Tuple[str, str]] = []
    for class_name in sorted(os.listdir(data_root)):
        class_dir = os.path.join(data_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        crops_dir = os.path.join(class_dir, "cropped")
        if not os.path.isdir(crops_dir):
            continue
        for f in sorted(os.listdir(crops_dir)):
            p = os.path.join(crops_dir, f)
            if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                items.append((p, class_name))
    return items


def find_mask_for(img_path: str, masks_dirname: str = "masks", cutouts_dirname: str = "cutouts") -> np.ndarray | None:
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
    bg = np.zeros_like(img_rgb)
    bg[..., 0] = gray_bg[0]
    bg[..., 1] = gray_bg[1]
    bg[..., 2] = gray_bg[2]
    m = mask.astype(bool)
    out = bg.copy()
    out[m] = img_rgb[m]
    return out


class AlbEvalDS(Dataset):
    """
    Dataset для валидации эмбеддера.
    
    Похож на AlbDS из обучающего скрипта, но упрощён для инференса.
    """
    def __init__(self, items: List[Tuple[str, str]], cls2id: Dict[str, int], aug, crop_mode: str = "bbox", mix_mask_p: float = 0.0, masks_dirname: str = "masks"):
        self.items = items
        self.cls2id = cls2id
        self.aug = aug
        self.crop_mode = crop_mode
        self.mix_mask_p = float(mix_mask_p)
        self.masks_dirname = masks_dirname

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int):
        import random
        p, c = self.items[i]
        img = np.array(Image.open(p).convert("RGB"))
        use_mask = False
        if self.crop_mode == "mask":
            use_mask = True
        elif self.crop_mode == "mix":
            use_mask = (random.random() < self.mix_mask_p)
        if use_mask:
            m = find_mask_for(p, masks_dirname=self.masks_dirname)
            if m is not None and m.shape[:2] == img.shape[:2]:
                img = apply_mask_np(img, m)
        out = self.aug(image=img)
        x = out["image"]
        y = self.cls2id[c]
        return x, y


class EmbedNet(nn.Module):
    """
    Нейросеть для извлечения эмбеддингов (такая же, как в обучающем скрипте).
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
    ArcFace margin product (такой же, как в обучающем скрипте).
    """
    def __init__(self, in_dim: int, n_classes: int, s: float = 30.0, m: float = 0.30):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, in_dim))
        nn.init.xavier_uniform_(self.W)
        self.s, self.m = s, m
    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        W = F.normalize(self.W, dim=1)
        cos = F.linear(z, W)
        one_hot = F.one_hot(y, num_classes=W.size(0)).float().to(z.device)
        return self.s * (cos - one_hot * self.m)


def build_transforms(img_size: int):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import inspect

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    pad_value = (114, 114, 114)

    def has_param(transform_cls, param: str) -> bool:
        try:
            return param in inspect.signature(transform_cls.__init__).parameters
        except Exception:
            return False

    pad_kwargs = {"min_height": img_size, "min_width": img_size, "border_mode": 0}
    if has_param(A.PadIfNeeded, "border_value"):
        pad_kwargs["border_value"] = pad_value
    elif has_param(A.PadIfNeeded, "value"):
        pad_kwargs["value"] = pad_value

    val_aug = A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(**pad_kwargs),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    return val_aug


def load_splits(splits_path: str, fallback_items: List[Tuple[str, str]]):
    """
    Загружает сохранённые сплиты данных.
    
    Args:
        splits_path: Путь к splits.json файлу
        fallback_items: Запасные данные при ошибке загрузки
        
    Returns:
        Кортеж (train_items, val_items, test_items)
    """
    train_items: List[Tuple[str, str]] = []
    val_items: List[Tuple[str, str]] = []
    test_items: List[Tuple[str, str]] = []
    try:
        if splits_path and os.path.isfile(splits_path):
            with open(splits_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            def from_rec(lst: List[Dict[str, Any]]):
                return [(d["path"], d["class"]) for d in lst]
            train_items = from_rec(data.get("train", []))
            val_items = from_rec(data.get("val", []))
            test_items = from_rec(data.get("test", []))
        else:
            raise FileNotFoundError("splits_path not provided or file not found")
    except Exception:
        test_items = fallback_items
    return train_items, val_items, test_items


def evaluate(args):
    """
    Основная функция валидации.
    
    Выполняет полный цикл валидации: загрузка модели → инференс →
    вычисление метрик → сохранение результатов.
    
    Args:
        args: Объект argparse с параметрами валидации
    """
    log = setup_logging(args.out_dir)
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Определяем путь к splits.json: либо из аргумента, либо рядом с чекпоинтом
    splits_path = args.splits_path
    if not splits_path:
        # Пытаемся найти рядом с чекпоинтом
        ckpt_dir = os.path.dirname(args.resume)
        cand = os.path.join(ckpt_dir, "splits.json") if ckpt_dir else ""
        if cand and os.path.isfile(cand):
            splits_path = cand
        else:
            # Пытаемся в out_dir
            cand2 = os.path.join(args.out_dir, "splits.json")
            if os.path.isfile(cand2):
                splits_path = cand2

    # 2) Загружаем сплиты либо используем fallback из data_root
    all_items_scan = discover_items_only_cropped(args.data_root)
    if args.merge_screwdrivers:
        all_items_scan = [(p, MERGE_MAP.get(c, c)) for (p, c) in all_items_scan]
    train_items, val_items, test_items = load_splits(splits_path, all_items_scan)

    # 3) Загружаем чекпоинт и извлекаем cls2id если есть
    assert os.path.isfile(args.resume), f"Checkpoint not found: {args.resume}"
    ckpt = torch.load(args.resume, map_location=device)

    cls2id: Dict[str, int] = {}
    if isinstance(ckpt, dict) and "cls2id" in ckpt and isinstance(ckpt["cls2id"], dict):
        # Используем mapping из обучения для строгого соответствия порядку классов
        cls2id = {str(k): int(v) for k, v in ckpt["cls2id"].items()}
    else:
        # Строим из сплитов (устойчиво к объединённым названиям классов)
        classes_from_splits = sorted(list({c for _, c in (val_items + test_items + train_items)}))
        if not classes_from_splits:
            # Фолбэк на сканирование каталога
            classes_from_splits = sorted(list({c for _, c in all_items_scan}))
        cls2id = {c: i for i, c in enumerate(classes_from_splits)}

    id2cls = {v: k for k, v in cls2id.items()}

    val_aug = build_transforms(args.img_size)

    pin = torch.cuda.is_available()
    dl_val = None
    if args.do_val and val_items:
        dl_val = DataLoader(AlbEvalDS(val_items, cls2id, val_aug, crop_mode="bbox", masks_dirname=args.masks_dirname),
                            batch_size=args.batch, shuffle=False, num_workers=args.workers,
                            pin_memory=pin, persistent_workers=pin and args.workers > 0)
    dl_test = None
    if args.do_test and test_items:
        dl_test = DataLoader(AlbEvalDS(test_items, cls2id, val_aug, crop_mode="bbox", masks_dirname=args.masks_dirname),
                             batch_size=args.batch, shuffle=False, num_workers=args.workers,
                             pin_memory=pin, persistent_workers=pin and args.workers > 0)

    model = EmbedNet(dim=args.emb_dim, model_name=args.model_id, pool="token",
                     freeze_backbone=True, img_size=args.img_size).to(device)
    arc = ArcMarginProduct(args.emb_dim, n_classes=len(cls2id)).to(device)
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt, strict=False)
    if isinstance(ckpt, dict) and "arc" in ckpt:
        try:
            arc.load_state_dict(ckpt["arc"], strict=False)
        except Exception:
            pass
    model.eval(); arc.eval()

    metrics_jsonl = os.path.join(args.out_dir, "val_metrics.jsonl")
    append_jsonl(metrics_jsonl, {"event": "val_start", "timestamp": time.time(), "resume": args.resume})

    def run_eval(dl, split_name: str):
        if dl is None:
            return
        log.info(f"[{split_name.upper()}] Начинаю оценку...")
        correct = 0; total = 0
        y_true: List[int] = []
        y_prob: List[np.ndarray] = []
        with torch.no_grad():
            pbar = tqdm(dl, desc=f"Eval {split_name}", leave=False)
            for x, y in pbar:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    z = model(x)
                    logits = arc(z, y)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item(); total += y.size(0)
                probs = torch.softmax(logits.detach().float(), dim=1).cpu().numpy()
                y_prob.append(probs)
                y_true.extend(y.detach().cpu().tolist())
                if total > 0:
                    pbar.set_postfix({"acc": f"{correct/max(1,total):.3f}"})
            pbar.close()
        acc = correct / max(1, total)
        try:
            y_prob_arr = np.concatenate(y_prob, axis=0) if y_prob else np.zeros((0, len(cls2id)))
            y_true_arr = np.array(y_true, dtype=int)
            y_pred_arr = np.argmax(y_prob_arr, axis=1) if y_prob_arr.size else np.array([], dtype=int)
            f1m = f1_score(y_true_arr, y_pred_arr, average='macro') if y_prob_arr.size else float('nan')
            aucm = roc_auc_score(y_true_arr, y_prob_arr, multi_class='ovr', average='macro') if y_prob_arr.size else float('nan')
            per_prec, per_rec, per_f1, per_sup = precision_recall_fscore_support(
                y_true_arr, y_pred_arr, labels=list(range(len(cls2id))), average=None, zero_division=0
            ) if y_prob_arr.size else (np.array([]), np.array([]), np.array([]), np.array([]))
            per_auc_list: List[float] = []
            if y_prob_arr.size:
                for k in range(len(cls2id)):
                    try:
                        y_true_bin = (y_true_arr == k).astype(int)
                        if y_true_bin.max() == y_true_bin.min():
                            raise ValueError("single-class")
                        per_auc_list.append(float(roc_auc_score(y_true_bin, y_prob_arr[:, k])))
                    except Exception:
                        per_auc_list.append(float('nan'))
            per_auc = np.array(per_auc_list, dtype=float) if y_prob_arr.size else np.array([])
        except Exception:
            f1m, aucm = float('nan'), float('nan')
            per_prec = per_rec = per_f1 = per_sup = per_auc = np.array([])
        log.info(f"[{split_name.upper()}] acc={acc:.3f} f1={f1m:.3f} auc={aucm:.3f}")
        try:
            rec = {
                "event": f"{split_name}_end",
                "timestamp": time.time(),
                "metrics": {
                    "acc": float(acc),
                    "f1_macro": float(f1m),
                    "auc_macro": float(aucm),
                },
                f"{split_name}_per_class": {
                    id2cls[i]: {
                        "precision": float(per_prec[i]),
                        "recall": float(per_rec[i]),
                        "f1": float(per_f1[i]),
                        "auc": (float(per_auc[i]) if i < per_auc.size and not np.isnan(per_auc[i]) else None),
                        "support": int(per_sup[i]),
                    } for i in range(len(per_f1))
                } if np.size(per_f1) else {},
            }
            append_jsonl(metrics_jsonl, rec)
        except Exception:
            pass

    if dl_val is not None:
        run_eval(dl_val, "val")
    if dl_test is not None:
        run_eval(dl_test, "test")

    if args.do_test and args.vis_test_n > 0 and test_items:
        log.info("[TEST] Визуализации до/после (bbox vs mask)")
        vis_dir = os.path.join(args.out_dir, "vis_test")
        os.makedirs(vis_dir, exist_ok=True)
        count_saved = 0
        for (p, c) in test_items[:args.vis_test_n]:
            try:
                with Image.open(p).convert("RGB") as im:
                    img_np = np.array(im)
                Image.fromarray(img_np).save(os.path.join(vis_dir, f"{os.path.splitext(os.path.basename(p))[0]}_bbox.jpg"))
                m = find_mask_for(p, masks_dirname=args.masks_dirname)
                if m is not None and m.shape[:2] == img_np.shape[:2]:
                    Image.fromarray(m.astype(np.uint8) * 255).save(os.path.join(vis_dir, f"{os.path.splitext(os.path.basename(p))[0]}_mask.png"))
                    Image.fromarray(apply_mask_np(img_np, m)).save(os.path.join(vis_dir, f"{os.path.splitext(os.path.basename(p))[0]}_masked.jpg"))
                count_saved += 1
            except Exception:
                continue
        append_jsonl(metrics_jsonl, {"event": "test_vis", "timestamp": time.time(), "saved": int(count_saved)})

    append_jsonl(metrics_jsonl, {"event": "val_end", "timestamp": time.time()})
    log.info("✅ Валидация завершена.")


def parse_args():
    """
    Парсинг аргументов командной строки.
    
    Примеры запуска:
    # Валидация на val сплите
    python lct-dino-3-val.py --data_root /path/to/dataset --out_dir /path/to/val_results \
        --splits_path /path/to/splits.json --resume /path/to/checkpoint_best.pth --do_val
    
    # Тестирование на test сплите с визуализациями
    python lct-dino-3-val.py --data_root /path/to/dataset --out_dir /path/to/test_results \
        --splits_path /path/to/splits.json --resume /path/to/checkpoint_best.pth --do_test --vis_test_n 100
    
    # Полная оценка (валидация + тест + визуализации)
    python lct-dino-3-val.py --data_root /path/to/dataset --out_dir /path/to/full_eval \
        --splits_path /path/to/splits.json --resume /path/to/checkpoint_best.pth --do_val --do_test
    """
    p = argparse.ArgumentParser(description="Валидация DINOv3 EmbedNet (ArcFace)")
    p.add_argument("--data_root", type=str, required=True, help="Корневая папка dataset с 'cropped'")
    p.add_argument("--out_dir", type=str, required=True, help="Папка для логов/метрик валидации")
    p.add_argument("--splits_path", type=str, default="", help="Путь к splits.json из трейна (если пусто, ищем рядом с чекпоинтом)")
    p.add_argument("--resume", type=str, required=True, help="Путь к чекпойнту (checkpoint_*.pth)")
    p.add_argument("--model_id", type=str, default="vit_large_patch16_dinov3.lvd1689m")
    p.add_argument("--emb_dim", type=int, default=128)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--masks_dirname", type=str, default="masks")
    p.add_argument("--do_val", action="store_true")
    p.add_argument("--do_test", action="store_true")
    p.add_argument("--vis_test_n", type=int, default=64, help="Сколько тестовых изображений сохранить для визуализации")
    p.add_argument("--merge_screwdrivers", action="store_true", help="Объединять классы отверток (+, -, крест) в один при сканировании датасета (fallback)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
