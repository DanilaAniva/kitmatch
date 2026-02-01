#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Валидация предразметки (predictions_final-3.json) по COCO-разметке (result.json).

Делает следующее:
1) Нормализует названия классов из предразметки до канонических имён из COCO.
2) Фильтрует предсказания по score_metric >= SCORE_THR и исключает 'unknown'.
3) Матчит предсказания и GT по IoU >= IOU_THR (greedy по score).
4) Считает TP/FP/FN и precision/recall/F1 per-class и per-image.
5) Сохраняет:
   - validation/predictions_normalized.json (копия предсказаний с нормализованными именами классов)
   - validation/validation_report.json (итоги валидации: сводка и по каждому изображению)
"""

import json
import os
import re
import math
import unicodedata as ud
from collections import defaultdict


# Пути
PRED_JSON = "/home/ubuntu/diabert/dataset/predrazmetka_dashi/predictions_final-3.json"
COCO_JSON = "/home/ubuntu/diabert/dataset/razmetka/result.json"
OUT_DIR   = "/home/ubuntu/diabert/dataset/predrazmetka_dashi/validation"

# Порог по предсказаниям и IoU
SCORE_THR = 0.56
IOU_THR   = 0.50


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_text(s: str) -> str:
    """Унификация строк: NFKC, нижний регистр, убираем лишние пробелы.
    Дополнительно удаляем ведущие номера классов и лишние символы-разделители,
    оставляя +, - и ¾, т.к. они информативны для названий.
    """
    if s is None:
        return ""
    s = ud.normalize("NFKC", s)
    # убрать лидирующий номер класса: "^\d+\s*"
    s = re.sub(r"^\s*\d+\s*", "", s)
    s = s.strip().lower()
    # унифицируем кавычки и слэши, а также специфичные символы
    s = s.replace('"', '"')
    s = s.replace('/', ' ')
    # Более надежная замена для 'й' на 'й'
    s = s.replace('й', 'й') # 'и' + combining breve (U+0306)
    s = s.replace('ой', 'ой') # 'о' + 'и' + combining breve
    s = s.replace('3⁄4', '¾') # Consistent representation of three-quarters
    s = re.sub(r"\s+", " ", s) # Замена множественных пробелов на один
    return s


def strong_normalize(s: str) -> str:
    """Более сильная нормализация для сопоставления: убираем все, кроме букв/цифр/+- и ¾."""
    s = normalize_text(s)
    keep = []
    for ch in s:
        if ch.isalnum() or ch in "+-¾": # Keep alphanumeric, +, -, ¾
            keep.append(ch)
    return "".join(keep)


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def greedy_match(preds, gts, iou_thr=0.5):
    """Гриди-маппинг предсказаний к GT по IoU. preds уже отсортированы по score убыв.
    preds: [{bbox:[x1,y1,x2,y2], score:float}]
    gts:   [[x1,y1,x2,y2], ...]
    Возврат: tp, fp, fn
    """
    used = [False] * len(gts)
    tp = 0
    fp = 0
    for p in preds:
        best_iou = 0.0
        best_j = -1
        for j, g in enumerate(gts):
            if used[j]:
                continue
            iou = iou_xyxy(p["bbox"], g)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_thr and best_j >= 0:
            used[best_j] = True
            tp += 1
        else:
            fp += 1
    fn = used.count(False)
    # print(f"  Greedy match results: TP={tp}, FP={fp}, FN={fn}")
    return tp, fp, fn


def main():
    ensure_dir(OUT_DIR)

    # Загрузка COCO
    with open(COCO_JSON, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Словарь image_base_name -> image_id и размеры
    imgid_by_name = {}
    size_by_imgid = {}
    for im in coco.get("images", []):
        base = os.path.basename(im.get("file_name", ""))
        # Дополнительная обработка для COCO, чтобы получить чистое имя файла (например, DSCN4994.JPG из 414f1ffd-DSCN4994.JPG)
        # Assuming filename is always like UUID-ORIGINALNAME.EXT
        match = re.search(r'-([^-]+\.JPG)$', base)
        if match:
            base = match.group(1)
        imgid_by_name[base] = im["id"]
        size_by_imgid[im["id"]] = (im.get("width", None), im.get("height", None))

    print(f"\nNumber of images in COCO GT: {len(imgid_by_name)}")
    print(f"First 5 images in COCO GT: {list(imgid_by_name.keys())[:5]}")

    # Категории COCO: id->name и нормализованные варианты
    catid_to_name = {c["id"]: c["name"] for c in coco.get("categories", [])}
    coco_names = list(catid_to_name.values())

    norm_to_coco = {}
    strong_to_coco = {}
    for name in coco_names:
        n = normalize_text(name)
        ns = strong_normalize(name)
        norm_to_coco[n] = name
        strong_to_coco[ns] = name

    # GT боксы по (image, class)
    gts_by_img_class = defaultdict(lambda: defaultdict(list))
    for ann in coco.get("annotations", []):
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        name = catid_to_name.get(cat_id, "")
        if not name:
            continue
        xyxy = xywh_to_xyxy(ann.get("bbox", [0, 0, 0, 0]))
        base_name = None
        # найдём base_name по image_id
        # обратный индекс: image_id -> (base_name)
        for k, v in imgid_by_name.items():
            if v == img_id:
                base_name = k
                break
        if base_name is None:
            continue
        gts_by_img_class[base_name][name].append(xyxy)

    print(f"Number of images with GT annotations: {len(gts_by_img_class)}")
    print(f"First 5 images with GT annotations: {list(gts_by_img_class.keys())[:5]}")

    # Загрузка предсказаний
    with open(PRED_JSON, "r", encoding="utf-8") as f:
        preds_raw = json.load(f)

    # Нормализация имён классов предсказаний -> имена из COCO
    preds_norm = []
    unknown_count = 0
    unmapped_names = set() # Changed from mapped_unknown to track actual unmapped names
    for rec in preds_raw:
        img_name = rec.get("image")
        out_dets = []
        for d in rec.get("detections", []):
            cls = d.get("class", "")
            if cls == "unknown":
                unknown_count += 1
                continue
            score = d.get("score_metric", 0.0)
            if score is None:
                score = 0.0
            if score < SCORE_THR:
                continue
            # нормализуем имя
            cls_n = normalize_text(cls)
            cls_s = strong_normalize(cls)
            coco_name = norm_to_coco.get(cls_n)
            if coco_name is None:
                coco_name = strong_to_coco.get(cls_s)
            # Если не нашли, как fallback — оставить как есть, но отметим
            if coco_name is None:
                unmapped_names.add(cls) # Add original unmapped name
                coco_name = cls_n # Keep the normalized form for now

            # Отладочный вывод для проверки маппинга
            # print(f"Mapping: '{cls}' -> normalized '{cls_n}' / strong '{cls_s}' -> COCO '{coco_name}'")

            out_dets.append({
                "bbox_xyxy": d.get("bbox_xyxy", [0, 0, 0, 0]),
                "class": coco_name,
                "score_metric": float(score),
                "score_vlm": float(d.get("score_vlm", 0.0))
            })
        preds_norm.append({"image": img_name, "detections": out_dets})

    # Сохраним нормализованные предсказания
    norm_path = os.path.join(OUT_DIR, "predictions_normalized.json")
    with open(norm_path, "w", encoding="utf-8") as f:
        json.dump(preds_norm, f, ensure_ascii=False, indent=2)

    # Подготовка для метрик: пер-класс суммарно и per-image
    class_totals = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    per_image = []

    # Итерация по изображениям, которые есть и в предсказаниях, и в COCO
    for rec in preds_norm:
        img_name = rec.get("image")
        if img_name not in gts_by_img_class:
            # нет GT для этого изображения — можно пропустить или считать все предсказания FP
            # Здесь пропустим, чтобы не искажать метрики.
            continue

        print(f"Processing image: {img_name}. Found in GT: {img_name in gts_by_img_class}. GT annotations for image: {len(gts_by_img_class.get(img_name, {}).keys())} classes")

        dets = rec.get("detections", [])

        # Группируем предсказания по классу
        preds_by_class = defaultdict(list)
        for d in dets:
            cls = d.get("class", "")
            bbox = d.get("bbox_xyxy", [0, 0, 0, 0])
            score = d.get("score_metric", 0.0)
            preds_by_class[cls].append({"bbox": bbox, "score": float(score)})

        # Для полноты пройдём по объединению классов из GT и предиктов
        classes = set(gts_by_img_class[img_name].keys()) | set(preds_by_class.keys())
        img_report = {"image": img_name, "classes": {}}
        for cls in sorted(classes):
            gts = gts_by_img_class[img_name].get(cls, [])
            prs = preds_by_class.get(cls, [])
            # сортировка предсказаний по убыванию score
            prs = sorted(prs, key=lambda x: x.get("score", 0.0), reverse=True)
            tp, fp, fn = greedy_match(prs, gts, iou_thr=IOU_THR)

            # Отладочный вывод для TP, FP, FN по каждому классу в изображении
            print(f"  Image '{img_name}', Class '{cls}': TP={tp}, FP={fp}, FN={fn}")

            class_totals[cls]["tp"] += tp
            class_totals[cls]["fp"] += fp
            class_totals[cls]["fn"] += fn
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            img_report["classes"][cls] = {
                "tp": tp, "fp": fp, "fn": fn,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "preds": len(prs),
                "gts": len(gts)
            }
        per_image.append(img_report)

    # Сводка по классам
    class_summary = {}
    for cls, agg in class_totals.items():
        tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        class_summary[cls] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4)
        }

    # Сохраняем отчёт
    report = {
        "settings": {
            "score_threshold": SCORE_THR,
            "iou_threshold": IOU_THR,
            "predictions": PRED_JSON,
            "ground_truth": COCO_JSON
        },
        "mapping_stats": {
            "unknown_skipped": unknown_count,
            "unmapped_names_after_norm": list(sorted(list(unmapped_names))) # Save actual unmapped names
        },
        "class_summary": class_summary,
        "per_image": per_image
    }

    out_path = os.path.join(OUT_DIR, "validation_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Готово. Нормализованные предсказания: {norm_path}")
    print(f"Готово. Отчёт валидации: {out_path}")


if __name__ == "__main__":
    main()



