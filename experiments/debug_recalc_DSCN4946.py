import os
import json
import math
import unicodedata
from typing import List, Dict


EXP_DIR = "/home/ubuntu/diabert/dataset/predrazmetka_dashi/dinov3-sam2-gdino_10_try"
IMAGE_STEM = "DSCN4946"
PRED_JSON = os.path.join(EXP_DIR, "predictions.json")
UNIQUE_REASSIGN_MIN_SCORE = 0.0  # держим синхронно с пайплайном


def canon(name: str) -> str:
    try:
        return unicodedata.normalize("NFC", str(name)).strip()
    except Exception:
        return str(name).strip()


def assign_unique_classes_clone(det_entries: List[Dict]) -> List[Dict]:
    if not det_entries:
        return []

    def best_score(idx: int) -> float:
        r = det_entries[idx].get("ranked") or []
        if r:
            try:
                return float(r[0].get("score", det_entries[idx].get("score_metric", 0.0)))
            except Exception:
                return float(det_entries[idx].get("score_metric", 0.0))
        return float(det_entries[idx].get("score_metric", 0.0))

    class_to_indices: Dict[str, List[int]] = {}
    for i, d in enumerate(det_entries):
        cname = canon(d.get("class", "") or "")
        class_to_indices.setdefault(cname, []).append(i)

    assigned_indices = set()
    used_classes = set()
    result = [dict(x) for x in det_entries]

    for cname, idxs in class_to_indices.items():
        if not idxs:
            continue
        best_idx = max(idxs, key=best_score)
        if cname:
            used_classes.add(cname)
        assigned_indices.add(best_idx)

    remaining = [i for i in range(len(det_entries)) if i not in assigned_indices]
    remaining.sort(key=best_score, reverse=True)
    kept_indices = set(assigned_indices)

    for i in remaining:
        ranked = det_entries[i].get("ranked") or []
        picked = False
        for cand in ranked:
            cname = canon(cand.get("name"))
            try:
                cscore = float(cand.get("score", 0.0))
            except Exception:
                cscore = 0.0
            if cname and (cname not in used_classes) and math.isfinite(cscore) and (cscore >= UNIQUE_REASSIGN_MIN_SCORE):
                result[i]["class"] = cname
                result[i]["score_metric"] = cscore
                used_classes.add(cname)
                kept_indices.add(i)
                picked = True
                break
        if not picked:
            # удалить объект, чтобы не было дублей
            pass

    final = [result[i] for i in sorted(kept_indices)]
    return final


def main():
    dets = []
    if os.path.exists(PRED_JSON):
        with open(PRED_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Находим запись по DSCN4946 (с любым расширением)
        rec = None
        for r in data:
            fn = os.path.splitext(r.get("image", ""))[0]
            if fn == IMAGE_STEM:
                rec = r
                break
        if rec is not None:
            dets = rec.get("detections", [])

    if dets:
        print("--- BEFORE RECALC (from predictions.json) ---")
        for i, d in enumerate(dets):
            print(f"{i:02d}: class={canon(d.get('class',''))!r} score={d.get('score_metric')} vlm={d.get('vlm_class')} bbox={d.get('bbox_xyxy')}")
        classes = [canon(d.get('class','')) for d in dets]
        summary = {}
        for c in classes:
            summary[c] = summary.get(c, 0) + 1
        print("Classes count before:", summary)
    else:
        print("No predictions.json or no entry for image; proceeding with debug-only inspection.")

    # Перерасчёт
    if dets:
        dets_re = assign_unique_classes_clone(dets)
        print("\n--- AFTER RECALC ---")
        for i, d in enumerate(dets_re):
            print(f"{i:02d}: class={canon(d.get('class',''))!r} score={d.get('score_metric')} bbox={d.get('bbox_xyxy')}")
        classes2 = [canon(d.get('class','')) for d in dets_re]
        summary2 = {}
        for c in classes2:
            summary2[c] = summary2.get(c, 0) + 1
        print("Classes count after:", summary2)

    # Сверка с debug similarities
    debug_dir = os.path.join(EXP_DIR, "debug", IMAGE_STEM)
    if os.path.isdir(debug_dir):
        crops = sorted([p for p in os.listdir(debug_dir) if p.startswith("crop_")])
        print(f"\nFound {len(crops)} debug crops")
        for c in crops:
            sp = os.path.join(debug_dir, c, "similarities.json")
            if not os.path.exists(sp):
                continue
            with open(sp, "r", encoding="utf-8") as f:
                sj = json.load(f)
            best = sj.get("best", {})
            allc = sj.get("all", [])
            print(f"{c}: best={canon(best.get('name',''))!r} score={best.get('score')} path={best.get('path')}")
            print("  top5:")
            for r in allc[:5]:
                print(f"    {canon(r.get('name',''))!r}: {r.get('score')}")
    else:
        print(f"No debug dir: {debug_dir}")


if __name__ == "__main__":
    main()


