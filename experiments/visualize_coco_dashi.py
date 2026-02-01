#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для визуализации разметки COCO с масками сегментации и bounding boxes
"""

import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import os
import shutil

def load_coco_data(json_path):
    """Загрузка данных COCO из JSON файла"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def polygon_to_mask(polygon, height, width):
    """Преобразование полигона в маску"""
    mask = np.zeros((height, width), dtype=np.uint8)
    # Полигон в формате [x1, y1, x2, y2, ...]
    points = np.array(polygon).reshape(-1, 2).astype(np.int32)
    cv2.fillPoly(mask, [points], 1)
    return mask

def visualize_predictions(pred_json_path, images_dir, limit_images=10, viz_subdir_name='visualizations_pred'):
    """
    Визуализация предразметки (bbox в формате xyxy) для первых N изображений.

    Args:
        pred_json_path: путь к JSON с предразметкой (список: {image, detections})
        images_dir: папка с исходными изображениями
        limit_images: ограничение по количеству визуализируемых изображений
        viz_subdir_name: имя подпапки для сохранения визуализаций рядом с JSON
    """
    # Загрузка данных предразметки
    with open(pred_json_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    base_dir = os.path.dirname(pred_json_path)
    viz_dir = os.path.join(base_dir, viz_subdir_name)
    os.makedirs(viz_dir, exist_ok=True)

    processed_count = 0

    for item in predictions:
        if processed_count >= limit_images:
            break

        image_name = item.get('image')
        detections = item.get('detections', [])

        if not image_name:
            continue

        image_path = os.path.join(images_dir, image_name)
        if not os.path.exists(image_path):
            print(f"Изображение не найдено в источнике: {image_path}")
            continue

        # Копируем исходник рядом с JSON (в его папку)
        try:
            shutil.copy2(image_path, os.path.join(base_dir, image_name))
        except Exception as e:
            print(f"Не удалось скопировать {image_name}: {e}")

        # Загрузка и подготовка изображения
        image = cv2.imread(image_path)
        if image is None:
            print(f"Ошибка чтения изображения: {image_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        # Цвета по классам
        class_names = [det.get('class', 'unknown') for det in detections]
        unique_classes = list(dict.fromkeys(class_names))  # сохраняем порядок появления
        colors = plt.cm.Set3(np.linspace(0, 1, max(len(unique_classes), 1)))
        class_to_color = {cls: colors[i % len(colors)] for i, cls in enumerate(unique_classes)}

        # Фигура
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.imshow(image)

        # Отрисовка bbox
        for det in detections:
            bbox = det.get('bbox_xyxy')
            cls_name = det.get('class', 'unknown')
            score = det.get('score_metric', None)
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            w = max(0.0, float(x2) - float(x1))
            h = max(0.0, float(y2) - float(y1))
            color = class_to_color.get(cls_name, (1.0, 0.0, 0.0, 1.0))

            rect = patches.Rectangle((x1, y1), w, h, linewidth=3,
                                     edgecolor=color, facecolor='none', alpha=0.9)
            ax.add_patch(rect)

            label = cls_name if score is None else f"{cls_name} {score:.2f}"
            ax.text(x1, max(0, y1 - 10), label, fontsize=12, color='white',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.85),
                    fontweight='bold')

        # Оформление
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.axis('off')
        plt.title(f'Предразметка: {image_name}', fontsize=16, fontweight='bold', pad=20)

        # Легенда по классам
        if unique_classes:
            legend_elements = [patches.Patch(color=class_to_color[c], label=c) for c in unique_classes]
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=10)

        plt.tight_layout()

        # Сохранение файла визуализации
        output_filename = f"viz_{os.path.splitext(image_name)[0]}.jpg"
        output_full_path = os.path.join(viz_dir, output_filename)
        plt.savefig(output_full_path, dpi=300, bbox_inches='tight')
        print(f"Визуализация сохранена: {output_full_path}")
        plt.close()

        processed_count += 1

    print(f"Готово. Создано визуализаций: {processed_count}. Исходники скопированы в: {base_dir}")

def main():
    """Основная функция"""
    # Пути к данным предразметки
    pred_json_path = '/home/ubuntu/diabert/dataset/predrazmetka_dashi/predictions_final-3.json'
    images_dir = '/home/ubuntu/diabert/dataset/dataset/Групповые для тренировки'

    limit_images = 10
    viz_subdir_name = 'visualizations_pred10'

    print("Начинаем создание визуализаций предразметки...")
    print(f"JSON предразметки: {pred_json_path}")
    print(f"Папка с изображениями: {images_dir}")
    print(f"Лимит изображений: {limit_images}")

    # Проверка существования
    if not os.path.exists(pred_json_path):
        print(f"Ошибка: JSON предразметки не найден: {pred_json_path}")
        return
    if not os.path.exists(images_dir):
        print(f"Ошибка: Папка с изображениями не найдена: {images_dir}")
        return

    try:
        visualize_predictions(pred_json_path, images_dir, limit_images=limit_images, viz_subdir_name=viz_subdir_name)
        print("Визуализации предразметки успешно созданы!")
    except Exception as e:
        print(f"Ошибка при создании визуализаций: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
