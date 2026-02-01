import torch, torchvision, torchaudio
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torchaudio:", torchaudio.__version__)
print("cuda:", torch.version.cuda)
import numpy, scipy, sklearn
print("numpy:", numpy.__version__)
print("scipy:", scipy.__version__)
print("sklearn:", sklearn.__version__)


import os, glob, json, random, math
import numpy as np
import contextlib
from PIL import Image, ImageDraw

BASE_DIR = "/kaggle/working"

IMG_DIR      = "/kaggle/input/lct-low/LCT/Групповые для тренировки"  # групповые фото
CROPS_TRAIN  = "/kaggle/input/lct-low/LCT"                           # кропы для обучения (подпапки-классы)
CROPS_REF    = "/kaggle/input/lct-ref/REF"                           # эталоны (8–16 на класс)

OUT_VLM      = os.path.join(BASE_DIR, "out_vlm")      # детекции от GroundingDINO
OUT_METRIC   = os.path.join(BASE_DIR, "out_metric")   # финальные визуалки и JSON
BEST_PATH    = os.path.join(BASE_DIR, "embedder_arcface_best.pth")
PROTO_PATH   = os.path.join(BASE_DIR, "prototypes.npz")
PRED_VLM     = os.path.join(OUT_VLM, "predictions_vlm.json")
PRED_FINAL   = os.path.join(OUT_METRIC, "predictions_final.json")     # финальные визуалки и JSON

for d in [IMG_DIR, CROPS_TRAIN, CROPS_REF, OUT_VLM, OUT_METRIC]:
    os.makedirs(d, exist_ok=True)

# Гиперпараметры
IMG_SIZE = 224
BATCH    = 64
EPOCHS   = 15
LR       = 1e-3
SEED     = 42
NUM_WORKERS = 2
# Пороговые пар-ры для финальной классификации
ALPHA        = 0.5   # смешивание centroid и kNN
TOPK         = 5
THR_UNKNOWN  = 0.10  # минимальный cos до прототипа
THR_MARGIN   = 0.03  # отрыв top1-top2
IOU_THR      = 0.55  # для class-wise NMS

## Обучим эмбеддер: DinoV2 + ArcFace (обучение)
# Модель
MODEL_NAME = "vit_base_patch14_dinov2.lvd142m"  # или "vit_base_patch14_dinov2.lvd142m"
EMB_DIM    = 128
FREEZE_BACKBONE = True      # 1-й этап: учим голову
# ==== IMPORTS & DEVICE ====
import glob, json, random, math
import numpy as np
from PIL import Image, ImageDraw
import unicodedata as ud

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import timm

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(SEED)

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ==== DATASET & UTILS ====

def _norm(s: str) -> str:
    return ud.normalize("NFKC", s).strip().lower()

def list_items(root):
    assert os.path.isdir(root), f"Папка не найдена: {root}"
    SKIP_SUBSTR = [_norm("групповые"), _norm("тренировк"), _norm("линейк")]

    items, classes = [], []
    for c in sorted(os.listdir(root)):
        if c.startswith("."): 
            continue
        pth = os.path.join(root, c)
        if not os.path.isdir(pth):
            continue
        if any(s in _norm(c) for s in SKIP_SUBSTR):
            continue
        classes.append(c)
        for f in glob.glob(os.path.join(pth, "*")):
            if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp")):
                items.append((f, c))

    assert len(classes) > 0, "Не найдено ни одного класса в CROPS_TRAIN"
    cls2id = {c:i for i,c in enumerate(classes)}
    return items, cls2id

class ImgDS(Dataset):
    def __init__(self, items, cls2id, tfm):
        self.items, self.cls2id, self.tfm = items, cls2id, tfm
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p,c = self.items[i]
        with Image.open(p).convert("RGB") as im:
            x = self.tfm(im)
        return x, self.cls2id[c]
# ==== TRANSFORMS (timm) ====
_tmp = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0, global_pool="token")
data_cfg = timm.data.resolve_model_data_config(_tmp)
if IMG_SIZE is not None:
    data_cfg["input_size"] = (3, IMG_SIZE, IMG_SIZE)

tfm_train = timm.data.create_transform(**data_cfg, is_training=True)
tfm_val   = timm.data.create_transform(**data_cfg, is_training=False)

# ==== MODEL & ARC ====

class EmbedNet(nn.Module):
    def __init__(self, dim=128, model_name=MODEL_NAME, pool="token", freeze_backbone=False, img_size=IMG_SIZE):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool=pool,
            img_size=img_size  # согласуем с трансформами
        )
        # Разрешаем вход не ровно "родного" размера (без динамической пересемпловки pos-эмбедов)
        if hasattr(self.backbone, "patch_embed"):
            try:
                self.backbone.patch_embed.strict_img_size = False
            except Exception:
                pass
        # ВАЖНО: НЕ включаем dynamic_img_size — это и ломало форму
        # if hasattr(self.backbone, "dynamic_img_size"):
        #     self.backbone.dynamic_img_size = True  # УДАЛЕНО

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

    def forward(self, x):
        f = self.backbone(x)                 # [B, feat_dim]
        z = F.normalize(self.head(f), dim=1) # [B, dim]
        return z

class ArcMarginProduct(nn.Module):
    def __init__(self, in_dim, n_classes, s=30.0, m=0.30):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, in_dim))
        nn.init.xavier_uniform_(self.W)
        self.s, self.m = s, m
    def forward(self, z, y):
        W = F.normalize(self.W, dim=1)
        cos = F.linear(z, W)
        one_hot = F.one_hot(y, num_classes=W.size(0)).float().to(z.device)
        return self.s * (cos - one_hot*self.m)

# инициализация
model = EmbedNet(dim=EMB_DIM, freeze_backbone=FREEZE_BACKBONE, img_size=IMG_SIZE).to(device)
arc   = ArcMarginProduct(EMB_DIM, n_classes=1).to(device)  # переопределим n_classes ниже
ce    = nn.CrossEntropyLoss()


# --- Albumentations аугментации для эмбеддера ---
# pip: !pip -q install albumentations>=1.4.0  # (на Kaggle обычно уже есть)
import albumentations as A
from albumentations.pytorch import ToTensorV2

# mean/std берём из timm data_cfg, чтобы всё совпадало с бэкбоном
_MEAN = data_cfg.get('mean', (0.5, 0.5, 0.5))
_STD  = data_cfg.get('std',  (0.5, 0.5, 0.5))

IMG_PAD_VALUE = (114, 114, 114)  # фон стола/бумаги

train_aug = A.Compose([
    A.LongestMaxSize(max_size=IMG_SIZE),
    A.PadIfNeeded(IMG_SIZE, IMG_SIZE, border_mode=0, value=IMG_PAD_VALUE),
    A.ShiftScaleRotate(shift_limit=0.04, scale_limit=0.15, rotate_limit=35,
                       border_mode=0, value=IMG_PAD_VALUE, p=0.8),
    A.HorizontalFlip(p=0.5),
    A.Perspective(scale=(0.02, 0.06), p=0.2),
    A.ColorJitter(0.3, 0.3, 0.3, 0.05, p=0.7),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    A.CLAHE(clip_limit=(1, 3), p=0.1),
    A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
    A.MotionBlur(blur_limit=3, p=0.1),
    A.ImageCompression(quality_lower=60, quality_upper=95, p=0.2),
    A.CoarseDropout(max_holes=4,
                    max_height=int(0.25*IMG_SIZE), max_width=int(0.25*IMG_SIZE),
                    min_holes=1, min_height=16, min_width=16,
                    fill_value=IMG_PAD_VALUE, p=0.5),
    A.Normalize(mean=_MEAN, std=_STD),
    ToTensorV2(),
])

val_aug = A.Compose([
    A.LongestMaxSize(max_size=IMG_SIZE),
    A.PadIfNeeded(IMG_SIZE, IMG_SIZE, border_mode=0, value=IMG_PAD_VALUE),
    A.Normalize(mean=_MEAN, std=_STD),
    ToTensorV2(),
])

# --- Dataset, который применяет Albumentations ---
class AlbDS(torch.utils.data.Dataset):
    def __init__(self, items, cls2id, aug):
        self.items = items
        self.cls2id = cls2id
        self.aug = aug

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        p, c = self.items[i]
        # PIL -> numpy(BGR/RGB неважно для A, но мы используем RGB)
        img = np.array(Image.open(p).convert("RGB"))
        out = self.aug(image=img)
        x = out["image"]  # torch.Tensor [C,H,W], уже нормирован
        y = self.cls2id[c]
        return x, y

# маленький обёртчик, чтобы те же самые преобразования использовать
# для прототипов и инференса кропов: вызов как tfm(image_PIL) -> torch.Tensor
class AlbWrap:
    def __init__(self, aug): self.aug = aug
    def __call__(self, pil_img):
        arr = np.array(pil_img.convert("RGB"))
        return self.aug(image=arr)["image"]

# ==== DATA & TRAIN ====
items, cls2id = list_items(CROPS_TRAIN)
train_items, val_items = train_test_split(items, test_size=0.2,
                                          stratify=[c for _,c in items], random_state=SEED)

# пересоздаём arc с реальным числом классов
del arc
arc = ArcMarginProduct(EMB_DIM, n_classes=len(cls2id)).to(device)

pin = torch.cuda.is_available()
dl_train = DataLoader(AlbDS(train_items, cls2id, train_aug), batch_size=BATCH, shuffle=True,
                      num_workers=NUM_WORKERS, pin_memory=pin, persistent_workers=pin and NUM_WORKERS>0)
dl_val   = DataLoader(AlbDS(val_items,   cls2id, val_aug),   batch_size=BATCH, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=pin, persistent_workers=pin and NUM_WORKERS>0)

# оптимизатор
if FREEZE_BACKBONE:
    params = list(model.head.parameters()) + list(arc.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
else:
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": max(LR*0.2, 5e-5), "weight_decay": 0.05},
        {"params": model.head.parameters(),     "lr": LR,                "weight_decay": 1e-4},
        {"params": arc.parameters(),            "lr": LR,                "weight_decay": 1e-4},
    ])

scaler = torch.amp.GradScaler(device='cuda', enabled=torch.cuda.is_available())
autocast_ctx = (torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available())
                if torch.cuda.is_available()
                else contextlib.nullcontext())
best_acc = 0.0
os.makedirs(os.path.dirname(BEST_PATH), exist_ok=True)

for ep in range(1, EPOCHS+1):
    model.train(); arc.train()
    tot=0; cnt=0
    pbar = tqdm(dl_train, desc=f"Epoch {ep}/{EPOCHS}", leave=False)
    for x,y in pbar:
        x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast_ctx:
            z = model(x)
            logits = arc(z, y)
            loss = ce(logits, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        tot += loss.item()*x.size(0); cnt += x.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    model.eval(); arc.eval()
    correct=0; total=0
    with torch.no_grad():
        for x,y in dl_val:
            x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                z = model(x); logits = arc(z,y)
            pred = logits.argmax(1)
            correct += (pred==y).sum().item(); total += y.size(0)

    val_acc = correct / max(1,total)
    print(f"Epoch {ep:02d} | train_loss={tot/max(1,cnt):.4f} | val_acc={val_acc:.3f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            "model": model.state_dict(),
            "cls2id": cls2id,
            "epoch": ep,
            "val_acc": val_acc,
            "model_name": MODEL_NAME,
            "emb_dim": EMB_DIM,
            "freeze_backbone": FREEZE_BACKBONE,
            "data_cfg": data_cfg,        # <-- сохраняем конфиг трансформов!
            "img_size": IMG_SIZE         # <-- и размер
        }, BEST_PATH)
        print(f"🌟 Новый лучший чекпоинт сохранён: {BEST_PATH} (acc={val_acc:.3f})")

print("✅ Обучение завершено. Лучший acc =", best_acc)

# трансформы для инференса/прототипов
tfm_proto = AlbWrap(val_aug)   # те же нормализация/паддинг/resize
tfm_infer = tfm_proto

import os, torch
ckpt_path = "/kaggle/working/embedder_arcface_best.pth"
print("Файл существует:", os.path.exists(ckpt_path), "| размер, байт:", os.path.getsize(ckpt_path) if os.path.exists(ckpt_path) else 0)

if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print("Ключи в чекпойнте:", list(ckpt.keys()))
    print("Число классов:", len(ckpt["cls2id"]))

for cls, idx in ckpt["cls2id"].items():
    print(f"{idx:2d} → {cls}")

#  0 → 1 Отвертка «-»
#  1 → 10 Ключ рожковыйнакидной  ¾
#  2 → 11 Бокорезы
#  3 → 2 Отвертка «+»
#  4 → 3 Отвертка на смещенный крест
#  5 → 4 Коловорот
#  6 → 5 Пассатижи контровочные
#  7 → 6 Пассатижи
#  8 → 7 Шэрница
#  9 → 8 Разводной ключ
# 10 → 9 Открывашка для банок с маслом

## Получим эмбеддинги на примерах классов (центроиды + стеки эталонов)
# ==== PROTOTYPES ====
import numpy as np

ckpt = torch.load(BEST_PATH, map_location="cpu")
cls2id = ckpt["cls2id"]; id2cls = {v:k for k,v in cls2id.items()}

# создаём модель на CPU с тем же img_size, что в ckpt
proto_model = EmbedNet(EMB_DIM, model_name=ckpt.get("model_name", MODEL_NAME),
                       freeze_backbone=True, img_size=ckpt.get("img_size", IMG_SIZE))
proto_model.load_state_dict(ckpt["model"])
proto_model.eval()

centroids, stacks = {}, {}
with torch.no_grad():
    for c in sorted(os.listdir(CROPS_REF)):
        pth = os.path.join(CROPS_REF, c)
        if not os.path.isdir(pth): 
            continue
        vecs=[]
        for f in glob.glob(os.path.join(pth, "*")):
            if not f.lower().endswith((".jpg",".png",".jpeg",".bmp",".webp")):
                continue
            with Image.open(f).convert("RGB") as im:
                x = tfm_proto(im).unsqueeze(0)  # CPU
            z = proto_model(x).squeeze(0).numpy()
            z /= max(np.linalg.norm(z), 1e-12)
            vecs.append(z)
        if vecs:
            M = np.stack(vecs, axis=0).astype(np.float32)
            mu = M.mean(0); mu /= max(np.linalg.norm(mu), 1e-12)
            centroids[c] = mu
            stacks[c]    = M

np.savez_compressed(PROTO_PATH,
                    centroids=centroids, stacks=stacks, id2cls=id2cls, cls2id=cls2id)
print("✅ Сохранили", PROTO_PATH)


NPZ = "/kaggle/working/prototypes.npz"
npz = np.load(NPZ, allow_pickle=True)
centroids = npz["centroids"].item()
stacks    = npz["stacks"].item()
print("\nКлассов в prototypes.npz:", len(centroids))

# Классов в prototypes.npz: 11

## GroundingDino: детекция боксов по промптам (без обучения)

# ==== GroundingDINO: детекция по текстовым категориям ====
import transformers as hf
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from torchvision.ops import nms

print("transformers:", hf.__version__)
print("torch:", torch.__version__)
use_cuda = torch.cuda.is_available()
gd_device = "cuda" if use_cuda else "cpu"

# Класс-лист
class_list = [
    "tool"
]
idx2class  = {i:c for i,c in enumerate(class_list)}
text_prompt = " . ".join(class_list) + " ."

MODEL_NAME_DINO = "IDEA-Research/grounding-dino-base"  # tiny быстрее; base точнее
processor  = AutoProcessor.from_pretrained(MODEL_NAME_DINO)
gdinomodel = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_NAME_DINO).to(gd_device).eval()

BOX_THR, TEXT_THR = 0.25, 0.25
VLM_IOU_THR, TOPK_PER_CLS = 0.55, 50
DO_VIS, BATCH_SIZE, SAVE_EVERY, RESUME = False, 2, 50, False

all_imgs = sorted([p for p in glob.glob(os.path.join(IMG_DIR, "*"))
                   if p.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp"))])
print(f"🖼️ найдено изображений: {len(all_imgs)}")

vlm_results = []
done_set = set()
pred_path = PRED_VLM  # <-- БЫЛО НЕОПРЕДЕЛЕНО, теперь есть

def save_json_atomic(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def to_tensor_safe(x, dtype=None, dev="cpu"):
    if isinstance(x, torch.Tensor):
        return x.to(device=dev, dtype=dtype) if (dtype or x.device.type != dev) else x
    return torch.tensor(x, dtype=dtype, device=dev)

def extract_label_names(res, idx2class):
    for k in ("text_labels","texts","phrases"):
        if k in res and res[k] is not None:
            try:
                return [str(s) for s in (res[k].tolist() if isinstance(res[k], torch.Tensor) else list(res[k]))]
            except Exception:
                pass
    for k in ("labels","label_ids","label_indices"):
        if k in res and res[k] is not None:
            raw = res[k]
            if isinstance(raw, torch.Tensor):
                idxs = raw.detach().cpu().to(torch.int64).tolist()
            else:
                seq = list(raw)
                if not seq: return []
                if isinstance(seq[0], str): return [str(s) for s in seq]
                try: idxs = [int(i) for i in seq]
                except Exception: return [str(x) for x in seq]
            return [idx2class[i] if 0 <= int(i) < len(idx2class) else None for i in idxs]
    return []

def run_batch(paths):
    imgs = [Image.open(p).convert("RGB") for p in paths]
    inputs = processor(images=imgs, text=[text_prompt]*len(imgs), return_tensors="pt").to(gd_device)
    with torch.no_grad():
        if use_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = gdinomodel(**inputs)
        else:
            outputs = gdinomodel(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs=outputs, input_ids=inputs.input_ids,
        box_threshold=BOX_THR, text_threshold=TEXT_THR,
        target_sizes=[im.size[::-1] for im in imgs],
    )

    batch_dets=[]
    for im,res in zip(imgs, results):
        boxes  = to_tensor_safe(res.get("boxes", []),  dtype=torch.float32, dev="cpu")
        scores = to_tensor_safe(res.get("scores", []), dtype=torch.float32, dev="cpu")
        labels_iter = extract_label_names(res, idx2class)

        n = min(len(boxes), len(scores), len(labels_iter))
        if n == 0:
            batch_dets.append([])
            continue
        boxes, scores, labels_iter = boxes[:n], scores[:n], labels_iter[:n]

        dets_by_cls = {}
        for bb, sc, name in zip(boxes, scores, labels_iter):
            if not name: continue
            dets_by_cls.setdefault(name, []).append((bb, float(sc)))

        dets = []
        for cls_name, arr in dets_by_cls.items():
            b = torch.stack([a[0] if isinstance(a[0], torch.Tensor) else torch.tensor(a[0], dtype=torch.float32) for a in arr], dim=0)
            s = torch.tensor([a[1] for a in arr], dtype=torch.float32)
            if use_cuda: b,s = b.to("cuda"), s.to("cuda")
            if len(s) > TOPK_PER_CLS:
                topk = torch.topk(s, TOPK_PER_CLS).indices
                b, s = b[topk], s[topk]
            if b.numel()==0: continue
            keep = nms(b, s, VLM_IOU_THR)
            b, s = b[keep].to("cpu"), s[keep].to("cpu")
            for bb, sc in zip(b, s):
                x1,y1,x2,y2 = [float(v) for v in bb.tolist()]
                dets.append({"class": cls_name, "score_vlm": float(sc), "bbox_xyxy": [x1,y1,x2,y2]})
        batch_dets.append(dets)

    for im in imgs:
        try: im.close()
        except Exception: pass

    return batch_dets

buf=[]
for i in tqdm(range(0, len(all_imgs), BATCH_SIZE), desc="GroundingDINO"):
    chunk = all_imgs[i:i+BATCH_SIZE]
    try:
        dets_list = run_batch(chunk)
    except Exception as e:
        print(f"⚠️ ошибка в run_batch [{i}:{i+BATCH_SIZE}]: {e}")
        continue

    if DO_VIS:
        for p, dets in zip(chunk, dets_list):
            try:
                with Image.open(p).convert("RGB") as im:
                    d = ImageDraw.Draw(im)
                    for det in dets:
                        x1,y1,x2,y2 = det["bbox_xyxy"]
                        d.rectangle([x1,y1,x2,y2], outline=(0,255,0), width=3)
                        d.text((max(0,x1+3), max(0,y1-14)), f'{det["class"]}:{det["score_vlm"]:.2f}', fill=(0,255,0))
                    im.save(os.path.join(OUT_VLM, os.path.splitext(os.path.basename(p))[0] + "_vlm.jpg"))
            except Exception as e:
                print("⚠️ визуализация:", e)

    for p, dets in zip(chunk, dets_list):
        rec = {"image": os.path.basename(p), "detections": dets}
        vlm_results.append(rec); buf.append(rec)

    if len(buf) >= SAVE_EVERY:
        save_json_atomic(pred_path, vlm_results)
        buf.clear()

if buf or not os.path.exists(pred_path):
    save_json_atomic(pred_path, vlm_results)

print("✅ GroundingDINO готово:", pred_path)


#  GroundingDINO готово: /kaggle/working/out_vlm/predictions_vlm.json
## Метрическая переклассификация + class-wise NMS
from torchvision.ops import nms
from torchvision import transforms
# загрузка эмбеддера и прототипов
ckpt = torch.load(os.path.join(BASE_DIR,"embedder_arcface_best.pth"), map_location=device)
clf_model = EmbedNet(128).to(device)
clf_model.load_state_dict(ckpt["model"])
clf_model.eval()

npz = np.load(os.path.join(BASE_DIR,"prototypes.npz"), allow_pickle=True)
# ассерт для прототипов
assert len(centroids) > 0, "centroids пустой — пересоздай prototypes.npz из CROPS_REF"
for c, M in stacks.items():
    if M is None or len(M) == 0:
        print(f"⚠️ В классе '{c}' нет эталонов в stacks — пропущу его")
# отбросим классы без стеков
centroids = {c:v for c,v in centroids.items() if c in stacks and len(stacks[c])>0}
stacks    = {c:stacks[c] for c in centroids.keys()}
assert len(centroids) > 0, "После фильтрации нет ни одного класса в прототипах"

centroids = npz["centroids"].item()
stacks    = npz["stacks"].item()

centroid_t = {c: torch.tensor(v).float().to(device) for c,v in centroids.items()}
stacks_t   = {c: torch.tensor(v).float().to(device) for c,v in stacks.items()}

tfm_infer = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

def classify_crop(crop):
    x = tfm_infer(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        z = clf_model(x)[0]     # [128], L2-норм
    scores = {}
    for c in centroids.keys():
        s_centroid = F.cosine_similarity(z, centroid_t[c], dim=0)
        stk = stacks_t[c]
        s_knn = torch.topk(torch.mv(stk, z), k=min(TOPK, stk.shape[0]))[0].mean()
        scores[c] = ALPHA*s_centroid.item() + (1-ALPHA)*s_knn.item()
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (c1,s1),(c2,s2) = ordered[0], ordered[1] if len(ordered)>1 else (("none",-1))
    if s1 < THR_UNKNOWN or (s1 - s2) < THR_MARGIN:
        return "unknown", s1
    return c1, s1

def classwise_nms(dets, iou_thr=IOU_THR):
    # dets: class, score_metric, bbox_xyxy
    out=[]
    by_cls={}
    for d in dets:
        by_cls.setdefault(d["class"], []).append(d)
    for c, arr in by_cls.items():
        if len(arr)==1 or c=="unknown":
            out += arr
            continue
        boxes  = torch.tensor([d["bbox_xyxy"] for d in arr]).float()
        scores = torch.tensor([d["score_metric"] for d in arr]).float()
        keep = nms(boxes, scores, iou_thr).tolist()
        out += [arr[i] for i in keep]
    return out

with open(os.path.join(OUT_VLM,"predictions_vlm.json"),"r",encoding="utf-8") as f:
    vlm_preds = json.load(f)

merged=[]
for rec in vlm_preds:
    img_name = rec["image"]
    img_path = os.path.join(IMG_DIR, img_name)
    if not os.path.exists(img_path):
        print("skip missing", img_path); continue
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    dets=[]
    for d in rec["detections"]:
        x1,y1,x2,y2 = d["bbox_xyxy"]
        # лёгкое расширение бокса на 10% чтобы захватить края
        w,h = x2-x1, y2-y1
        cx,cy = x1+w/2, y1+h/2
        x1n = max(0, cx - 0.55*w); y1n = max(0, cy - 0.55*h)
        x2n = min(img.width,  cx + 0.55*w); y2n = min(img.height, cy + 0.55*h)
        crop = img.crop((x1n,y1n,x2n,y2n)) # вырезаем инструмент

        label, conf = classify_crop(crop) # прогоняем через классификатор
        dets.append({
            "bbox_xyxy":[float(x1n),float(y1n),float(x2n),float(y2n)],
            "class":label,
            "score_metric":float(conf),
            "score_vlm":float(d.get("score_vlm",0.0)),
            "vlm_class":d.get("class","")
        })

    dets = classwise_nms(dets, IOU_THR) # применяем nms

    # визуализация
    vis = img.copy(); draw = ImageDraw.Draw(vis)
    for d in dets:
        x1,y1,x2,y2 = d["bbox_xyxy"]
        color = (0,255,0) if d["class"]!="unknown" else (255,0,0)
        draw.rectangle([x1,y1,x2,y2], outline=color, width=3)
        draw.text((x1+3,y1+3), f'{d["class"]}:{d["score_metric"]:.2f}', fill=color)
    stem = os.path.splitext(img_name)[0]
    vis.save(os.path.join(OUT_METRIC, f"{stem}_final.jpg"))

    merged.append({"image": img_name, "detections": dets})

with open(os.path.join(OUT_METRIC,"predictions_final.json"),"w",encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

print("✅ Финальные детекции:", os.path.join(OUT_METRIC,"predictions_final.json"))


# ✅ Финальные детекции: /kaggle/working/out_metric/predictions_final.json