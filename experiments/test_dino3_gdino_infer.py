import os, json, argparse, glob, time
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import timm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


# Defaults based on your environment
DEFAULT_IMAGE = "/home/ubuntu/diabert/dataset/predrazmetka_dashi/dinov3-sam2-gdino_20251001_140104/DSCN4946.JPG"
DEFAULT_GALLERY = "/home/ubuntu/diabert/dataset/crops_of_every_tool/Кропнутые инструменты все"
GROUNDING_DINO_MODEL = "rziga/mm_grounding_dino_large_all"
GROUNDING_PROMPT = "tool"
BOX_THR = 0.25
TEXT_THR = 0.25

# Strict DINOv3 (no dinov2 fallback)
TIMM_DINOV3_ID = "vit_large_patch16_dinov3.lvd1689m"


class DinoV3Embedder:
    def __init__(self, model_id: str, device: torch.device, use_fp16: bool = False):
        self.device = device
        self.model = timm.create_model(model_id, pretrained=True, num_classes=0).to(self.device).eval()
        if self.device.type == "cuda" and use_fp16:
            self.model = self.model.half()
        data_cfg = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**data_cfg, is_training=False)

    @torch.no_grad()
    def embed(self, img: Image.Image) -> torch.Tensor:
        if img.mode != "RGB":
            img = img.convert("RGB")
        t_img = self.transforms(img).unsqueeze(0).to(self.device)
        feats = self.model(t_img)
        feats = F.normalize(feats, dim=1)
        return feats.squeeze(0).detach().cpu().float()


class GroundingDINODetector:
    def __init__(self, device: torch.device):
        self.processor = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_DINO_MODEL).to(device).eval()
        self.prompt_txt = " . ".join([GROUNDING_PROMPT]) + " ."
        self.box_thr = float(BOX_THR)
        self.text_thr = float(TEXT_THR)
        self.device = device

    @torch.no_grad()
    def detect(self, image: Image.Image) -> List[dict]:
        inputs = self.processor(images=[image], text=[self.prompt_txt], return_tensors="pt").to(self.device)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=(self.device.type=="cuda")):
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs, input_ids=inputs.input_ids, threshold=self.box_thr, text_threshold=self.text_thr, target_sizes=[image.size[::-1]]
        )
        res = results[0]
        boxes = res.get("boxes", [])
        scores = res.get("scores", [])
        dets = []
        for i in range(min(len(boxes), len(scores))):
            bb = boxes[i].detach().float().cpu().tolist()
            sc = float(scores[i].detach().cpu())
            dets.append({"bbox_xyxy": [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])], "score_vlm": sc, "class": GROUNDING_PROMPT})
        return dets


def load_gallery(embedder: DinoV3Embedder, gdir: str, limit: int | None = 64) -> Tuple[List[str], torch.Tensor]:
    paths = sorted([p for p in glob.glob(os.path.join(gdir, "*")) if p.lower().endswith((".png",".jpg",".jpeg",".webp",".bmp"))])
    if limit:
        paths = paths[:max(1, int(limit))]
    names, vecs = [], []
    for p in paths:
        try:
            with Image.open(p) as im:
                z = embedder.embed(im.convert("RGB"))
            names.append(os.path.splitext(os.path.basename(p))[0])
            vecs.append(z.cpu())
        except Exception:
            continue
    if not vecs:
        raise RuntimeError("Gallery is empty or failed to embed any image")
    return names, torch.stack(vecs, dim=0)


def match(z: torch.Tensor, gallery_vecs: torch.Tensor, gallery_names: List[str]) -> Tuple[str, float]:
    z = torch.nan_to_num(z.float(), nan=0.0, posinf=0.0, neginf=0.0)
    g = torch.nan_to_num(gallery_vecs.float(), nan=0.0, posinf=0.0, neginf=0.0)
    z = F.normalize(z.view(-1), dim=0, eps=1e-12)
    g = F.normalize(g, dim=1, eps=1e-12)
    sims = torch.mv(g, z)
    k = int(torch.argmax(sims).item())
    return gallery_names[k], float(sims[k].item())


def run(image_path: str, gallery_dir: str, device_str: str) -> dict:
    device = torch.device(device_str if device_str else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    t0 = time.perf_counter()
    embedder = DinoV3Embedder(TIMM_DINOV3_ID, device)
    det = GroundingDINODetector(device)
    names, G = load_gallery(embedder, gallery_dir)
    init_s = time.perf_counter() - t0

    img = Image.open(image_path).convert("RGB")
    dets = det.detect(img)
    results = []
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["bbox_xyxy"]]
        x1 = max(0, min(x1, img.width - 1)); x2 = max(0, min(x2, img.width))
        y1 = max(0, min(y1, img.height - 1)); y2 = max(0, min(y2, img.height))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img.crop((x1, y1, x2, y2)).convert("RGB")
        z = embedder.embed(crop)
        name, score = match(z, G, names)
        results.append({
            "class": str(name),
            "confidence": float(score),
            "x_min": int(x1), "y_min": int(y1), "x_max": int(x2), "y_max": int(y2),
            "detector_score": float(d.get("score_vlm", 0.0)),
        })
    return {
        "init_seconds": round(init_s, 3),
        "bboxes": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--gallery", default=DEFAULT_GALLERY)
    ap.add_argument("--device", default="")
    args = ap.parse_args()
    out = run(args.image, args.gallery, args.device)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


