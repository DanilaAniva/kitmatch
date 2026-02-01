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

def visualize_coco_annotations(json_path, images_dir, output_path='visualization.jpg'):
    """
    Создание визуализации разметки COCO
    
    Args:
        json_path: путь к JSON файлу с аннотациями
        images_dir: папка с изображениями
        output_path: путь для сохранения результата
    """
    # Загрузка данных
    coco_data = load_coco_data(json_path)
    
    # Создание словарей для быстрого поиска
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Обработка каждого изображения
    for image_info in coco_data['images']:
        image_id = image_info['id']
        image_filename = image_info['file_name']
        
        # Попытка найти изображение в папке images
        image_base_name = os.path.basename(image_filename)
        image_path = os.path.join(images_dir, image_base_name)
        
        if not os.path.exists(image_path):
            print(f"Изображение не найдено: {image_path}")
            continue
            
        # Загрузка изображения
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        # Создание фигуры
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.imshow(image)
        
        # Найти все аннотации для данного изображения
        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
        
        # Цвета для разных категорий
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        category_colors = {cat_id: colors[i] for i, cat_id in enumerate(categories.keys())}
        
        for ann in annotations:
            category_id = ann['category_id']
            category_name = categories[category_id]
            color = category_colors[category_id]
            
            # Отрисовка маски сегментации
            if 'segmentation' in ann and ann['segmentation']:
                for segmentation in ann['segmentation']:
                    if len(segmentation) >= 6:  # Минимум 3 точки
                        # Создание маски
                        mask = polygon_to_mask(segmentation, height, width)
                        
                        # Создание цветной маски с прозрачностью
                        colored_mask = np.zeros((height, width, 4))
                        colored_mask[:, :, :3] = color[:3]
                        colored_mask[:, :, 3] = mask * 0.4  # Прозрачность
                        
                        # Наложение маски
                        ax.imshow(colored_mask, alpha=0.6)
                        
                        # Отрисовка контура
                        points = np.array(segmentation).reshape(-1, 2)
                        polygon = Polygon(points, linewidth=2, edgecolor=color, 
                                        facecolor='none', alpha=0.8)
                        ax.add_patch(polygon)
            
            # Отрисовка bounding box
            if 'bbox' in ann:
                x, y, w, h = ann['bbox']
                rect = patches.Rectangle((x, y), w, h, linewidth=3, 
                                       edgecolor=color, facecolor='none', alpha=0.8)
                ax.add_patch(rect)
                
                # Добавление подписи категории
                ax.text(x, y-10, category_name, fontsize=12, color='white',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                       fontweight='bold')
        
        # Настройка отображения
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.axis('off')
        
        # Заголовок с именем файла
        plt.title(f'Визуализация разметки: {image_base_name}', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Создание легенды
        legend_elements = [patches.Patch(color=category_colors[cat_id], 
                                       label=categories[cat_id]) 
                          for cat_id in categories.keys()]
        ax.legend(handles=legend_elements, loc='upper right', 
                 bbox_to_anchor=(1.15, 1), fontsize=10)
        
        plt.tight_layout()
        
        # Сохранение результата
        output_filename = f"visualization_{image_base_name.split('.')[0]}.jpg"
        output_full_path = os.path.join(os.path.dirname(json_path), output_filename)
        plt.savefig(output_full_path, dpi=300, bbox_inches='tight')
        print(f"Визуализация сохранена: {output_full_path}")
        
        plt.close()  # Закрытие фигуры для освобождения памяти

def main():
    """Основная функция"""
    # Пути к файлам
    json_path = '/home/ubuntu/diabert/dataset/razmetka/result.json'
    images_dir = '/home/ubuntu/diabert/dataset/razmetka/images'
    
    print("Начинаем создание визуализации разметки COCO...")
    print(f"JSON файл: {json_path}")
    print(f"Папка с изображениями: {images_dir}")
    
    # Проверка существования файлов
    if not os.path.exists(json_path):
        print(f"Ошибка: JSON файл не найден: {json_path}")
        return
    
    if not os.path.exists(images_dir):
        print(f"Ошибка: Папка с изображениями не найдена: {images_dir}")
        return
    
    try:
        visualize_coco_annotations(json_path, images_dir)
        print("Визуализация успешно создана!")
    except Exception as e:
        print(f"Ошибка при создании визуализации: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
