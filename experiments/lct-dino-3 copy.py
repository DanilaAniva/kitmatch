import os, sys, json, math, random, argparse, time
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import timm
from sklearn.metrics import f1_score, roc_auc_score


def setup_logging(out_dir: str):
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_str(s: str) -> str:
    try:
        import unicodedata as ud
        return ud.normalize("NFKC", s).strip().lower()
    except Exception:
        return s.strip().lower()


def discover_items(data_root: str) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
    """Ищем классы как подпапки в data_root и изображения в подкаталоге 'cropped'.

    Возвращает:
      - список пар (путь_к_изображению, имя_класса)
      - словарь {класс: id}
    """
    assert os.path.isdir(data_root), f"Папка не найдена: {data_root}"
    SKIP_SUBSTR = [normalize_str("групповые"), normalize_str("тренировк"), normalize_str("линейк")]

    items: List[Tuple[str, str]] = []
    classes: List[str] = []
    for class_name in sorted(os.listdir(data_root)):
        if class_name.startswith("."):
            continue
        class_dir = os.path.join(data_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        if any(s in normalize_str(class_name) for s in SKIP_SUBSTR):
            continue

        crops_dir = os.path.join(class_dir, "cropped")
        search_dir = crops_dir if os.path.isdir(crops_dir) else class_dir
        has_any = False
        for f in sorted(os.listdir(search_dir)):
            p = os.path.join(search_dir, f)
            if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                items.append((p, class_name))
                has_any = True
        if has_any:
            classes.append(class_name)

    assert len(classes) > 0, "Не найдено ни одного класса (папки с картинками)"
    cls2id = {c: i for i, c in enumerate(sorted(classes))}
    return items, cls2id


class AlbWrap:
    def __init__(self, aug):
        self.aug = aug
    def __call__(self, pil_img: Image.Image):
        arr = np.array(pil_img.convert("RGB"))
        return self.aug(image=arr)["image"]


class AlbDS(Dataset):
    def __init__(self, items: List[Tuple[str, str]], cls2id: Dict[str, int], aug):
        self.items = items
        self.cls2id = cls2id
        self.aug = aug
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i: int):
        p, c = self.items[i]
        img = np.array(Image.open(p).convert("RGB"))
        out = self.aug(image=img)
        x = out["image"]
        y = self.cls2id[c]
        return x, y


class EmbedNet(nn.Module):
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


def train(args):
    # Создаем папку с номером рана
    run_number = 1
    base_out_dir = args.out_dir
    while os.path.exists(os.path.join(base_out_dir, f"run_{run_number:03d}")):
        run_number += 1
    args.out_dir = os.path.join(base_out_dir, f"run_{run_number:03d}")
    
    log = setup_logging(args.out_dir)
    log.info(f"Запуск #{run_number}, папка: {args.out_dir}")
    set_seed(args.seed)

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

    mean = tuple(data_cfg.get("mean", (0.5, 0.5, 0.5)))
    std = tuple(data_cfg.get("std", (0.5, 0.5, 0.5)))
    pad_value = (114, 114, 114)

    train_aug = A.Compose([
        A.LongestMaxSize(max_size=args.img_size),
        A.PadIfNeeded(args.img_size, args.img_size, border_mode=0, value=pad_value),
        A.ShiftScaleRotate(shift_limit=0.04, scale_limit=0.15, rotate_limit=35,
                           border_mode=0, value=pad_value, p=0.8),
        A.HorizontalFlip(p=0.5),
        A.Perspective(scale=(0.02, 0.06), p=0.2),
        A.ColorJitter(0.3, 0.3, 0.3, 0.05, p=0.7),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.CLAHE(clip_limit=(1, 3), p=0.1),
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
        A.MotionBlur(blur_limit=3, p=0.1),
        A.ImageCompression(quality_lower=60, quality_upper=95, p=0.2),
        A.CoarseDropout(max_holes=4,
                        max_height=int(0.25 * args.img_size), max_width=int(0.25 * args.img_size),
                        min_holes=1, min_height=16, min_width=16,
                        fill_value=pad_value, p=0.5),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    val_aug = A.Compose([
        A.LongestMaxSize(max_size=args.img_size),
        A.PadIfNeeded(args.img_size, args.img_size, border_mode=0, value=pad_value),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    # Данные
    items, cls2id = discover_items(args.data_root)
    labels_all = [c for _, c in items]
    from sklearn.model_selection import train_test_split
    train_items, val_items = train_test_split(
        items, test_size=args.val_split, stratify=labels_all, random_state=args.seed
    )

    pin = torch.cuda.is_available()
    dl_train = DataLoader(AlbDS(train_items, cls2id, train_aug), batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=pin, persistent_workers=pin and args.workers > 0)
    dl_val = DataLoader(AlbDS(val_items, cls2id, val_aug), batch_size=args.batch, shuffle=False,
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
            except Exception:
                f1_tr, auc_tr = float('nan'), float('nan')
            # Метрики val
            try:
                y_prob_val_arr = np.concatenate(y_prob_val, axis=0) if y_prob_val else np.zeros((0, len(cls2id)))
                y_true_val_arr = np.array(y_true_val, dtype=int)
                y_pred_val_arr = np.argmax(y_prob_val_arr, axis=1) if y_prob_val_arr.size else np.array([], dtype=int)
                f1_val = f1_score(y_true_val_arr, y_pred_val_arr, average='macro') if y_prob_val_arr.size else float('nan')
                auc_val = roc_auc_score(y_true_val_arr, y_prob_val_arr, multi_class='ovr', average='macro') if y_prob_val_arr.size else float('nan')
            except Exception:
                f1_val, auc_val = float('nan'), float('nan')

            log.info(
                f"Epoch {ep:02d} | train_loss={tot/max(1,cnt):.4f} | "
                f"acc_tr={train_acc:.3f} f1_tr={f1_tr:.3f} auc_tr={auc_tr:.3f} | "
                f"acc_val={val_acc:.3f} f1_val={f1_val:.3f} auc_val={auc_val:.3f}"
            )

            # Сохранение чекпоинтов
            save_ckpt(last_ckpt_path, ep, best_acc)
            if (ep % max(1, args.save_every)) == 0:
                save_ckpt(os.path.join(args.out_dir, f"checkpoint_ep{ep:03d}.pth"), ep, best_acc)
            if val_acc > best_acc:
                best_acc = val_acc
                save_ckpt(best_ckpt_path, ep, best_acc)
                log.info(f"🌟 Новый лучший чекпоинт сохранён (acc={best_acc:.3f})")

    except KeyboardInterrupt:
        log.warning("⛔ Остановка по KeyboardInterrupt — сохраняю interrupt-чекпоинт...")
        save_ckpt(os.path.join(args.out_dir, "checkpoint_interrupt.pth"), ep if 'ep' in locals() else 0, best_acc)
    except Exception as e:
        log.exception(f"💥 Аварийная ошибка обучения: {e}")
        raise

    # Экспорт прототипов из всех доступных кропов (centroids и stacks)
    log.info("[PROTO] Считаю прототипы (centroids/stacks)...")
    tfm_proto = AlbWrap(val_aug)

    # Модель на CPU для инференса прототипов (экономия VRAM)
    proto_model = EmbedNet(args.emb_dim, model_name=args.model_id,
                           freeze_backbone=True, img_size=args.img_size)
    proto_model.load_state_dict(model.state_dict(), strict=False)
    proto_model.eval()

    centroids: Dict[str, np.ndarray] = {}
    stacks: Dict[str, np.ndarray] = {}
    with torch.no_grad():
        for c in sorted(cls2id.keys()):
            # берем все изображения класса из dataset
            class_dir = os.path.join(args.data_root, c)
            crops_dir = os.path.join(class_dir, "cropped")
            search_dir = crops_dir if os.path.isdir(crops_dir) else class_dir
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
            if vecs:
                M = np.stack(vecs, axis=0).astype(np.float32)
                mu = M.mean(0); mu /= max(np.linalg.norm(mu), 1e-12)
                centroids[c] = mu
                stacks[c] = M

    proto_path = os.path.join(args.out_dir, "prototypes.npz")
    np.savez_compressed(proto_path,
                        centroids=centroids, stacks=stacks,
                        cls2id=cls2id, id2cls={v: k for k, v in cls2id.items()})
    log.info(f"✅ Прототипы сохранены: {proto_path}")
    log.info(f"✅ Лучший чекпоинт: {best_ckpt_path if os.path.exists(best_ckpt_path) else last_ckpt_path}")


def parse_args():
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
    p.add_argument("--save_every", type=int, default=5,
                   help="Каждые N эпох сохранить именованный чекпоинт")
    p.add_argument("--resume", type=str, default="",
                   help="Путь к чекпоинту для продолжения (по умолчанию ищется checkpoint_last.pth в out_dir)")
    p.add_argument("--max_refs_per_class", type=int, default=None,
                   help="Максимум эталонов на класс при построении прототипов (None — без ограничения)")
    p.add_argument("--gpus", type=int, default=None, help="Число GPU для DataParallel (например, 2)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)


