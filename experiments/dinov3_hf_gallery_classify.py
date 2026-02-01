import os
import glob
import json
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel


# =============================
#  Конфигурация и токен HF
# =============================
# Токен Hugging Face: задайте переменную окружения HF_TOKEN или передайте hf_token в DinoV3HFEmbedder.
# Для gated-моделей: https://huggingface.co/settings/tokens

# Принудительно используем float32 (как в эталоне HF)
try:
    torch.set_default_dtype(torch.float32)
except Exception:
    pass


def _is_image_file(path: str) -> bool:
    lower = path.lower()
    return lower.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"))


def find_image_files(directory: str, recursive: bool = True, limit: Optional[int] = None) -> List[str]:
    """Ищет изображения в папке (включая вложенные, если recursive=True).

    Возвращает отсортированный список путей. Можно ограничить первыми N.
    """
    if not os.path.isdir(directory):
        return []
    pattern = "**/*" if recursive else "*"
    paths = [p for p in glob.glob(os.path.join(directory, pattern), recursive=recursive) if _is_image_file(p)]
    paths = sorted(paths)
    if limit is not None:
        paths = paths[: max(0, int(limit))]
    return paths


def open_image_rgb(path: str) -> Optional[Image.Image]:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def l2_normalize_torch(vec: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vec, ord=2, dim=dim, keepdim=True)
    norm = torch.clamp(norm, min=eps)
    return vec / norm


class DinoV3HFEmbedder:
    """Эмбеддер DINOv3 (Hugging Face), загрузка на одну GPU в FP16.

    По умолчанию использует большую модель ViT-7B/16 (facebook/dinov3-vit7b16-pretrain-lvd1689m).
    """

    def __init__(
        self,
        model_id: str = "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        device: Optional[str] = None,
        hf_token: Optional[str] = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type != "cuda":
            raise RuntimeError("CUDA не доступна. Нужна минимум одна GPU для запуска DINOv3.")

        self.model_id = model_id
        self.hf_token = hf_token or os.getenv("HF_TOKEN", None)

        # Загрузка процессора и модели
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_id, use_fast=True, token=self.hf_token
        )
        self.model = AutoModel.from_pretrained(
            self.model_id, low_cpu_mem_usage=True, token=self.hf_token
        )

        # Перенос на GPU и в float32 (как в эталоне HF)
        self.model.to(self.device).to(torch.float32).eval()

        # Небольшая информация
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"[INFO] Модель загружена: {self.model_id}")
            print(f"[INFO] Параметров: {total_params/1e9:.2f}B, устройство: {self.device}, dtype: float32")
        except Exception:
            print(f"[INFO] Модель загружена: {self.model_id} на {self.device} (float32)")

    @torch.inference_mode()
    def embed(self, image: Image.Image) -> torch.Tensor:
        """Возвращает L2-нормированный эмбеддинг (Tensor[D], float32, на CPU).

        Если у модели есть pooler_output, берём его. Иначе берём CLS-токен [0,0].
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        batch = self.processor(images=image, return_tensors="pt")
        # Перенос входов на устройство
        if hasattr(batch, "to"):
            batch = batch.to(self.device)
        else:
            batch = {k: v.to(self.device) for k, v in batch.items()}

        # Без autocast, в полном float32 (как в эталоне HF)
        outputs = self.model(**batch)

        if getattr(outputs, "pooler_output", None) is not None:
            vec = outputs.pooler_output[0]
        else:
            vec = outputs.last_hidden_state[0, 0]

        # Приводим к float32 на CPU и нормализуем
        vec = torch.nan_to_num(vec.float().detach().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
        vec = l2_normalize_torch(vec, dim=0)
        return vec

    @torch.inference_mode()
    def embed_many(self, images: List[Image.Image], batch_size: int = 1) -> torch.Tensor:
        """Эмбеддинг списка PIL изображений. Возвращает матрицу [N, D] на CPU (float32).

        Большие батчи могут быть полезны, но осторожно с памятью GPU.
        """
        if not images:
            return torch.empty((0, 0), dtype=torch.float32)
        all_vecs: List[torch.Tensor] = []
        for i in range(0, len(images), max(1, int(batch_size))):
            chunk = images[i : i + batch_size]
            batch = self.processor(images=[im.convert("RGB") for im in chunk], return_tensors="pt")
            if hasattr(batch, "to"):
                batch = batch.to(self.device)
            else:
                batch = {k: v.to(self.device) for k, v in batch.items()}
            # Без autocast, в полном float32
            outputs = self.model(**batch)
            if getattr(outputs, "pooler_output", None) is not None:
                vecs = outputs.pooler_output
            else:
                vecs = outputs.last_hidden_state[:, 0]
            vecs = torch.nan_to_num(vecs.float().detach().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
            vecs = l2_normalize_torch(vecs, dim=1)
            all_vecs.append(vecs)
        return torch.cat(all_vecs, dim=0)


def load_gallery_embeddings(
    embedder: DinoV3HFEmbedder,
    gallery_paths: List[str],
) -> Tuple[List[str], torch.Tensor]:
    """Загружает изображения-эталоны и считает их эмбеддинги.

    Возвращает:
      - список имён классов (имя файла без расширения)
      - матрицу эмбеддингов [N, D] (float32, CPU)
    """
    images: List[Image.Image] = []
    names: List[str] = []
    for p in gallery_paths:
        im = open_image_rgb(p)
        if im is None:
            print(f"[WARN] Пропуск эталона (не открыть): {p}")
            continue
        images.append(im)
        names.append(os.path.splitext(os.path.basename(p))[0])
    if not images:
        raise RuntimeError("Не удалось загрузить ни одного эталонного изображения")
    G = embedder.embed_many(images, batch_size=1)
    return names, G


def cosine_similarities_to_gallery(query_vec: torch.Tensor, gallery_matrix: torch.Tensor) -> torch.Tensor:
    """Косинусные сходства между вектором [D] и галереей [N,D]. Возвращает [N]."""
    if query_vec.dim() != 1:
        query_vec = query_vec.view(-1)
    query_vec = l2_normalize_torch(torch.nan_to_num(query_vec.float(), nan=0.0, posinf=0.0, neginf=0.0), dim=0)
    G = l2_normalize_torch(torch.nan_to_num(gallery_matrix.float(), nan=0.0, posinf=0.0, neginf=0.0), dim=1)
    sims = torch.mv(G, query_vec)
    return sims


def compare_query_with_gallery(
    embedder: DinoV3HFEmbedder,
    gallery_paths: List[str],
    query_image_path: str,
    print_scores: bool = True,
) -> Dict:
    """Сравнивает один запрос с эталонами и печатает скоры.

    Возвращает словарь с полями: names, scores, best_name, best_score.
    """
    query_image = open_image_rgb(query_image_path)
    if query_image is None:
        raise RuntimeError(f"Не удалось открыть запрос: {query_image_path}")

    class_names, gallery_matrix = load_gallery_embeddings(embedder, gallery_paths)
    query_vec = embedder.embed(query_image)
    scores = cosine_similarities_to_gallery(query_vec, gallery_matrix)

    # Сортировка по убыванию
    order = torch.argsort(scores, descending=True).tolist()
    best_index = int(order[0])
    best_name = class_names[best_index]
    best_score = float(scores[best_index])

    if print_scores:
        print("\n[РЕЗУЛЬТАТЫ] Косинусные сходства (чем больше, тем ближе):")
        for idx in order:
            nm = class_names[int(idx)]
            sc = float(scores[int(idx)])
            print(f"  - {nm}: {sc:.4f}")
        print(f"\n[ТОП] Предсказанный класс: '{best_name}', скор: {best_score:.4f}")

    return {
        "names": class_names,
        "scores": [float(scores[i]) for i in range(scores.shape[0])],
        "best_name": best_name,
        "best_score": best_score,
    }


# =============================
#  Пример использования в ноутбуке
# =============================
#
# from dinov3_hf_gallery_classify import (
#     DinoV3HFEmbedder,
#     find_image_files,
#     compare_query_with_gallery,
# )
#
# embedder = DinoV3HFEmbedder(
#     model_id="facebook/dinov3-vit7b16-pretrain-lvd1689m",
# )
#
# gallery_dir = "/home/ubuntu/diabert/dataset/crops_of_every_tool"
# gallery_paths = find_image_files(gallery_dir, recursive=True, limit=11)
# query_path = "/path/to/your/query_image.png"  # замените на свой путь
#
# result = compare_query_with_gallery(embedder, gallery_paths, query_path, print_scores=True)
# print(result)


def main() -> None:
    # Константы запуска (при необходимости поменяйте под себя)
    query_path = \
        "/home/ubuntu/diabert/dataset/predrazmetka_dashi/dinov3-sam2-gdino_10_try/debug/DSCN4946/crop_00/crop.png"
    gallery_dir = \
        "/home/ubuntu/diabert/dataset/crops_of_every_tool"
    top_k_gallery = 11

    print("[MAIN] Старт")
    print("[STAGE] Инициализация эмбеддера DINOv3 (HF, FP16, 1xGPU)...")
    embedder = DinoV3HFEmbedder(model_id="facebook/dinov3-vitl16-pretrain-lvd1689m")

    print("[STAGE] Поиск эталонных изображений в галерее...")
    gallery_paths = find_image_files(gallery_dir, recursive=True, limit=top_k_gallery)
    if not gallery_paths:
        raise SystemExit(f"В галерее не найдено изображений: {gallery_dir}")
    print(f"[OK] Найдено эталонов: {len(gallery_paths)}")

    print("[STAGE] Сравнение запроса с эталонами...")
    result = compare_query_with_gallery(embedder, gallery_paths, query_path, print_scores=True)

    # Сохранение результатов в JSON рядом с запросом
    print("[STAGE] Сохранение результатов в JSON...")
    out_dir = os.path.dirname(os.path.abspath(query_path))
    out_path = os.path.join(out_dir, "dinov3_hf_scores.json")

    names = result.get("names", [])
    scores = result.get("scores", [])
    name2score = {str(names[i]): float(scores[i]) for i in range(min(len(names), len(scores)))}

    payload = {
        "model_id": getattr(embedder, "model_id", ""),
        "query": query_path,
        "gallery_paths": gallery_paths,
        "names": names,
        "scores": scores,
        "name_to_score": name2score,
        "best_name": result.get("best_name"),
        "best_score": float(result.get("best_score", float("nan"))),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"✅ Готово: результаты сохранены в {out_path}")


if __name__ == "__main__":
    main()


