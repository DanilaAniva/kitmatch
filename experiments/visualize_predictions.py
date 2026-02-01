import os, json
from typing import Dict, List
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Настройки по умолчанию
PREDICTIONS_JSON = "/home/ubuntu/diabert/dataset/predrazmetka_dashi/dinov3-sam2-gdino_10_try/predictions.json"
IMAGES_DIR = "/home/ubuntu/diabert/dataset/pредrazmetka_dashi/ishodniki_10_dashi"
OUT_DIR = "/home/ubuntu/diabert/dataset/predrazmetka_dashi/dinov3-sam2-gdino_10_try/visualized_from_predictions"

os.makedirs(OUT_DIR, exist_ok=True)


def rle_to_mask(rle: Dict) -> np.ndarray:
    size = rle.get("size")
    counts = rle.get("counts")
    h, w = int(size[0]), int(size[1])
    flat = np.zeros(h * w, dtype=np.uint8)
    cur = 0
    val = 0
    for c in counts:
        c = int(c)
        if c > 0:
            if val == 1:
                flat[cur:cur+c] = 1
        cur += c
        val = 1 - val
    return flat.reshape(h, w).astype(bool)


def save_with_masks(img: Image.Image, dets: List[Dict], out_path: str) -> None:
    class_names = [d.get('class', 'unknown') for d in dets]
    unique_classes = list(dict.fromkeys(class_names))
    colors = plt.cm.Set3(np.linspace(0, 1, max(len(unique_classes), 1)))
    class_to_color = {cls: colors[i % len(colors)] for i, cls in enumerate(unique_classes)}

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.imshow(np.array(img))

    # Единый оверлей
    H, W = img.height, img.width
    overlay = np.zeros((H, W, 4), dtype=np.uint8)
    alpha_val = int(0.4 * 255)
    for d in dets:
        rle = d.get('mask_rle')
        if not rle:
            continue
        mask = rle_to_mask(rle)
        color = class_to_color.get(d.get('class', 'unknown'), (1.0, 0.0, 0.0, 1.0))
        rgb255 = (np.array(color[:3]) * 255).astype(np.uint8)
        overlay[mask, 0] = rgb255[0]
        overlay[mask, 1] = rgb255[1]
        overlay[mask, 2] = rgb255[2]
        overlay[..., 3][mask] = np.maximum(overlay[..., 3][mask], alpha_val)
    if np.any(overlay[..., 3] > 0):
        ax.imshow(overlay, interpolation='nearest')

    # BBoxes и подписи
    for d in dets:
        x1,y1,x2,y2 = d["bbox_xyxy"]
        w = max(0.0, float(x2) - float(x1))
        h = max(0.0, float(y2) - float(y1))
        cls_name = d.get('class', 'unknown')
        score = d.get('score_metric', None)
        color = class_to_color.get(cls_name, (1.0, 0.0, 0.0, 1.0))

        rect = patches.Rectangle((x1, y1), w, h, linewidth=3,
                                 edgecolor=color, facecolor='none', alpha=0.9)
        ax.add_patch(rect)

        label = cls_name if (score is None) else f"{cls_name} {score:.2f}"
        ax.text(x1, max(0, y1 - 10), label, fontsize=8, color='white',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.85),
                fontweight='bold')

    ax.set_xlim(0, img.width)
    ax.set_ylim(img.height, 0)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    with open(PREDICTIONS_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for rec in data:
        img_name = rec.get('image')
        img_path = os.path.join(IMAGES_DIR, img_name)
        if not os.path.exists(img_path):
            print('skip not found', img_path)
            continue
        img = Image.open(img_path).convert('RGB')
        dets = rec.get('detections', [])
        out_path = os.path.join(OUT_DIR, os.path.splitext(img_name)[0] + '_from_predictions.jpg')
        save_with_masks(img, dets, out_path)
        print('saved', out_path)


if __name__ == '__main__':
    main()
