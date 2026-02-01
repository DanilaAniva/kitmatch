"""
Инференс с GroundingDINO, SAM-2 и DINOv3 эмбеддером

Этот скрипт выполняет полный пайплайн детекции и классификации инструментов
на новых изображениях. Объединяет три модели для максимальной точности:

1. GroundingDINO находит bounding boxes инструментов по текстовому промпту
2. SAM-2 генерирует точные маски по найденным boxes
3. DINOv3 эмбеддер классифицирует обрезанные и замаскированные области

Основная логика:
1. Загружает галерею PNG эталонов инструментов и вычисляет их эмбеддинги
2. Для каждого входного изображения:
   - Запускает GroundingDINO детекцию
   - Генерирует маски через SAM-2  
   - Обрезает и маскирует найденные области
   - Сравнивает с галереей через косинусное сходство эмбеддингов
   - Присваивает класс по максимальному сходству
3. Сохраняет результаты в JSON и опционально визуализации

Модели:
- GroundingDINO (IDEA-Research/grounding-dino-base) для детекции
- SAM-2 (hiera_tiny по умолчанию) для сегментации  
- DINOv3 эмбеддер (обученный через lct-dino-3.py) для классификации
"""

import os, sys, glob, json, math, argparse
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except Exception:
    raise SystemExit("timm не установлен. Установите: pip install timm --upgrade")

try:
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
except Exception:
    raise SystemExit("transformers не установлен. Установите: pip install transformers --upgrade")

# SAM-2 (пытаемся импортировать из установленного пакета; если нет — из локального репозитория)
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except Exception:
    sam2_repo = "/home/ubuntu/sam2"
    if os.path.isdir(sam2_repo):
        sys.path.insert(0, sam2_repo)
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    else:
        raise SystemExit("SAM-2 не найден. Выполните: cd /home/ubuntu/sam2 && pip install -e '.[notebooks]'")


def parse_args():
    """
    Парсинг аргументов командной строки.
    
    Returns:
        Объект с параметрами инференса
    """
    p = argparse.ArgumentParser(description="Инференс: GroundingDINO -> SAM-2 маски -> сравнение с PNG-эталонами эмбеддером DINOv3")
    p.add_argument("--images", type=str, default="/home/ubuntu/diabert/dataset/dataset/Групповые для тренировки",
                   help="Папка с групповыми изображениями")
    p.add_argument("--gallery", type=str, default="/home/ubuntu/diabert/dataset/crops_of_every_tool/Кропнутые инструменты все",
                   help="Папка с PNG эталонами (по одному на класс или больше)")
    p.add_argument("--base-dir", type=str, default="/home/ubuntu/diabert/dataset/predrazmetka_dashi",
                   help="Папка для результатов")
    p.add_argument("--ckpt", type=str, default="/home/ubuntu/diabert/dataset/predrazmetka_dashi/embedder_arcface_best.pth",
                   help="Путь к чекпоинту эмбеддера DINOv3+ArcFace")
    p.add_argument("--dino-text", type=str, default="tool",
                   help="Текстовый промпт для GroundingDINO (через точку разделять несколько)")
    p.add_argument("--box-thr", type=float, default=0.25)
    p.add_argument("--text-thr", type=float, default=0.25)
    p.add_argument("--vlm-iou-thr", type=float, default=0.55)
    p.add_argument("--sam2-cfg", type=str, default="/home/ubuntu/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml",
                   help="Путь к YAML конфигу SAM-2")
    p.add_argument("--sam2-ckpt", type=str, default="/home/ubuntu/sam2/checkpoints/sam2.1_hiera_tiny.pt",
                   help="Путь к весам SAM-2")
    p.add_argument("--save-vis", action="store_true", help="Сохранять визуализации")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


args = parse_args()

BASE_DIR   = args.base_dir
OUT_VLM    = os.path.join(BASE_DIR, "out_vlm")
OUT_METRIC = os.path.join(BASE_DIR, "out_metric")
PRED_VLM   = os.path.join(OUT_VLM, "predictions_vlm.json")
PRED_FINAL = os.path.join(OUT_METRIC, "predictions_final.json")
os.makedirs(OUT_VLM, exist_ok=True)
os.makedirs(OUT_METRIC, exist_ok=True)


# ==== Эмбеддер (DINOv3) из чекпоинта обучения ====
ckpt = torch.load(args.ckpt, map_location="cpu")
MODEL_NAME = ckpt.get("model_name", "vit_base_patch14_dinov3.lvd142m")
IMG_SIZE   = int(ckpt.get("img_size", 224))
EMB_DIM    = int(ckpt.get("emb_dim", 128))

class EmbedNet(nn.Module):
    """
    Нейросеть для извлечения эмбеддингов (такая же, как в lct-dino-3.py).
    """
    def __init__(self, dim=EMB_DIM, model_name=MODEL_NAME, img_size=IMG_SIZE):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool="token", img_size=img_size)
        if hasattr(self.backbone, "patch_embed"):
            try:
                self.backbone.patch_embed.strict_img_size = False
            except Exception:
                pass
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, dim),
        )
    def forward(self, x):
        f = self.backbone(x)
        z = F.normalize(self.head(f), dim=1)
        return z

device = torch.device(args.device)
embedder = EmbedNet().to(device).eval()
embedder.load_state_dict(ckpt["model"], strict=True)

_tmp = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0, global_pool="token")
data_cfg = timm.data.resolve_model_data_config(_tmp)
data_cfg["input_size"] = (3, IMG_SIZE, IMG_SIZE)
tfm_infer = timm.data.create_transform(**data_cfg, is_training=False)


# ==== Галерея PNG эталонов (все файлы в папке) ====
def load_gallery_embeddings(gdir: str) -> List[Tuple[str, torch.Tensor]]:
    """
    Загружает галерею эталонных изображений и вычисляет их эмбеддинги.
    
    Args:
        gdir: Папка с PNG/JPG эталонами инструментов
        
    Returns:
        Список кортежей (имя_файла, эмбеддинг_тензор)
    """
    out = []
    files = sorted([p for p in glob.glob(os.path.join(gdir, "*")) if p.lower().endswith((".png",".jpg",".jpeg",".webp",".bmp"))])
    for fp in files:
        try:
            with Image.open(fp).convert("RGB") as im:
                x = tfm_infer(im).unsqueeze(0).to(device)
            with torch.no_grad():
                z = embedder(x)[0]
            out.append((os.path.basename(fp), z.detach().cpu()))
        except Exception as e:
            print("skip gallery", fp, e)
    if not out:
        raise SystemExit(f"В галерее нет валидных изображений: {gdir}")
    return out

gallery = load_gallery_embeddings(args.gallery)
gallery_feats = torch.stack([g[1] for g in gallery], dim=0)  # [N,D]
gallery_names = [g[0] for g in gallery]


# ==== GroundingDINO (HF) для детекции боксов ====
MODEL_NAME_DINO = "IDEA-Research/grounding-dino-base"
processor  = AutoProcessor.from_pretrained(MODEL_NAME_DINO)
gdinomodel = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_NAME_DINO).to(device).eval()

class_list = [s.strip() for s in args.dino_text.split('.') if s.strip()]
idx2class  = {i:c for i,c in enumerate(class_list)}
text_prompt = " . ".join(class_list) + " ."

def dino_detect(img_pil: Image.Image) -> List[Dict]:
    """
    Выполняет детекцию инструментов через GroundingDINO.
    
    Args:
        img_pil: Изображение PIL в RGB формате
        
    Returns:
        Список детекций с bbox и скорами
    """
    inputs = processor(images=[img_pil], text=[text_prompt], return_tensors="pt").to(device)
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type=="cuda")):
            outputs = gdinomodel(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs=outputs, input_ids=inputs.input_ids,
        box_threshold=args.box_thr, text_threshold=args.text_thr,
        target_sizes=[img_pil.size[::-1]],
    )
    res = results[0]
    boxes  = res.get("boxes", [])
    scores = res.get("scores", [])
    labels = res.get("labels", []) if "labels" in res else res.get("text_labels", [])
    out=[]
    for i in range(min(len(boxes), len(scores))):
        bb = boxes[i].detach().float().cpu().tolist()
        sc = float(scores[i].detach().cpu())
        out.append({"bbox_xyxy": [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])], "score_vlm": sc, "class": "tool"})
    return out


# ==== SAM-2 предиктор ====
sam2 = build_sam2(args.sam2_cfg, args.sam2_ckpt)
sam2 = sam2.to(device).eval()
sam2_pred = SAM2ImagePredictor(sam2)

def sam2_mask_for_box(img_np: np.ndarray, box_xyxy: List[float]) -> np.ndarray:
    """
    Генерирует маску объекта по bounding box через SAM-2.
    
    Args:
        img_np: Изображение в формате numpy [H, W, 3]
        box_xyxy: Координаты bounding box [x1, y1, x2, y2]
        
    Returns:
        Булева маска [H, W]
    """
    # box в XYXY; возвращаем bool-маску HxW
    sam2_pred.set_image(img_np)
    x1,y1,x2,y2 = box_xyxy
    b = np.array([x1,y1,x2,y2], dtype=np.float32)
    masks, ious, _ = sam2_pred.predict(box=b[None, :], multimask_output=True)
    if masks.shape[0] == 0:
        return np.zeros((img_np.shape[0], img_np.shape[1]), dtype=bool)
    # выбираем по максимальному IoU score
    best = int(np.argmax(ious.reshape(-1)))
    m = masks[best].astype(bool)
    return m


def apply_mask_and_crop(img_pil: Image.Image, mask: np.ndarray) -> Image.Image:
    """
    Обрезает изображение по tight bbox маски и применяет маску к альфа-каналу.
    
    Args:
        img_pil: Исходное RGB изображение
        mask: Булева маска [H, W]
        
    Returns:
        Обрезанное RGBA изображение с применённой маской
    """
    # обрезаем по маске tight bbox и применяем маску на прозрачность
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return img_pil
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    crop = img_pil.crop((x1, y1, x2+1, y2+1)).convert("RGBA")
    m_crop = mask[y1:y2+1, x1:x2+1]
    alpha = (m_crop.astype(np.uint8) * 255)
    rgba = np.array(crop)
    rgba[..., 3] = alpha
    return Image.fromarray(rgba)


def embed_image(pil_img: Image.Image) -> torch.Tensor:
    """
    Вычисляет эмбеддинг для изображения через обученный DINOv3.
    
    Обрабатывает RGBA изображения (конвертирует к RGB на белом фоне).
    
    Args:
        pil_img: Изображение PIL (может быть RGB или RGBA)
        
    Returns:
        Нормализованный эмбеддинг вектор [D] на device
    """
    # для прозрачных PNG конвертируем к RGB на белом фоне, затем tfm_infer
    if pil_img.mode == "RGBA":
        bg = Image.new("RGB", pil_img.size, (255, 255, 255))
        bg.paste(pil_img, mask=pil_img.split()[3])
        pil_img = bg
    x = tfm_infer(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        z = embedder(x)[0]
    return z.detach()


def match_to_gallery(z: torch.Tensor) -> Tuple[str, float]:
    """
    Находит наиболее похожий эталон в галерее по косинусному сходству.
    
    Args:
        z: эмбеддинг запроса [D] на device
        
    Returns:
        Кортеж (имя_лучшего_эталона, скор_сходства)
    """
    # z: [D] на device; gallery_feats: [N,D] на cpu -> перенесём на device для косинуса
    g = gallery_feats.to(z.device)
    sims = torch.mv(g, z)
    best_idx = int(torch.argmax(sims).item())
    return gallery_names[best_idx], float(sims[best_idx].item())


def main():
    """
    Главная функция инференса.
    
    Выполняет полный пайплайн: детекция → сегментация → классификация.
    
    Примеры запуска:
    # Базовый инференс с сохранением визуализаций
    python infer_sam2_dinov3.py --save-vis
    
    # Кастомные папки и модель
    python infer_sam2_dinov3.py --images /path/to/test/images --gallery /path/to/reference/tools \
        --ckpt /path/to/custom_embedder.pth --base-dir /path/to/output
    
    # Использование другой модели SAM-2 с кастомными порогами
    python infer_sam2_dinov3.py --sam2-cfg configs/sam2.1/sam2.1_hiera_b.yaml \
        --sam2-ckpt /path/to/sam2.1_hiera_base.pt --box-thr 0.3 --text-thr 0.3
    """
    all_imgs = sorted([p for p in glob.glob(os.path.join(args.images, "*")) if p.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp"))])
    results=[]
    for p in all_imgs:
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            print("skip", p, e); continue

        dets = dino_detect(img)
        if not dets:
            results.append({"image": os.path.basename(p), "detections": []})
            continue

        img_np = np.array(img)
        vis = img.copy()
        draw = ImageDraw.Draw(vis)
        out_dets=[]
        for d in dets:
            x1,y1,x2,y2 = d["bbox_xyxy"]
            mask = sam2_mask_for_box(img_np, [x1,y1,x2,y2])
            cut = apply_mask_and_crop(img, mask)
            z = embed_image(cut)
            best_name, best_sim = match_to_gallery(z)
            out_dets.append({
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "class": best_name,
                "score_metric": best_sim,
                "score_vlm": float(d.get("score_vlm", 0.0)),
                "vlm_class": d.get("class", "")
            })

            if args.save_vis:
                color = (0,255,0)
                draw.rectangle([x1,y1,x2,y2], outline=color, width=3)
                draw.text((x1+3,y1+3), f"{best_name}:{best_sim:.2f}", fill=color)

        results.append({"image": os.path.basename(p), "detections": out_dets})
        if args.save_vis:
            stem = os.path.splitext(os.path.basename(p))[0]
            vis.save(os.path.join(OUT_METRIC, f"{stem}_sam2_match.jpg"))

    with open(PRED_FINAL, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("✅ Готово. Результаты сохранены:", PRED_FINAL)


if __name__ == "__main__":
    main()


