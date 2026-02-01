#!/usr/bin/env python3
"""
Скрипт для конвертации датасета Label Studio в формат YOLO.

Конвертирует аннотации из формата Label Studio в формат YOLO v8/v11:
- Создает структуру папок train/val
- Конвертирует координаты из процентов в YOLO формат (нормализованные)
- Создает data.yaml конфигурационный файл
"""

import os
import json
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple
import yaml

def convert_ls_to_yolo(
    ls_json_path: str,
    images_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    seed: int = 42
) -> str:
    """
    Конвертирует датасет Label Studio в формат YOLO.

    Args:
        ls_json_path: путь к JSON файлу Label Studio
        images_dir: директория с изображениями
        output_dir: выходная директория для YOLO датасета
        train_ratio: доля данных для обучения
        val_ratio: доля данных для валидации
        seed: seed для воспроизводимости

    Returns:
        путь к созданному data.yaml файлу
    """

    # Загружаем данные Label Studio
    print(f"Загрузка данных из {ls_json_path}")
    with open(ls_json_path, 'r', encoding='utf-8') as f:
        ls_data = json.load(f)

    print(f"Найдено {len(ls_data)} аннотаций")

    # Собираем уникальные классы и создаем маппинг
    classes = set()
    for item in ls_data:
        if 'annotations' in item:
            for annotation in item['annotations']:
                if 'result' in annotation:
                    for result in annotation['result']:
                        if 'value' in result and 'rectanglelabels' in result['value']:
                            for label in result['value']['rectanglelabels']:
                                classes.add(label)

    # Функция нормализации названий классов для объединения дубликатов
    def normalize_class_name(name: str) -> str:
        """Нормализует название класса, объединяя похожие варианты"""
        name = name.strip()
        # Объединяем похожие названия ключей
        if name in ["Ключ рожковый/накидной ¾", "Ключ рожковыйнакидной ¾"]:
            return "Ключ рожковый/накидной ¾"
        # Можно добавить другие правила объединения здесь
        return name

    # Нормализуем названия классов
    normalized_classes = set()
    for cls in classes:
        normalized_classes.add(normalize_class_name(cls))

    class_list = sorted(list(normalized_classes))
    class_to_id = {cls: i for i, cls in enumerate(class_list)}

    print(f"Найдено {len(class_list)} классов:")
    for i, cls in enumerate(class_list):
        print(f"  {i}: {cls}")

    # Создаем структуру директорий
    output_path = Path(output_dir)
    images_train = output_path / "images" / "train"
    images_val = output_path / "images" / "val"
    labels_train = output_path / "labels" / "train"
    labels_val = output_path / "labels" / "val"

    for dir_path in [images_train, images_val, labels_train, labels_val]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Разделяем на train/val
    random.seed(seed)
    random.shuffle(ls_data)

    n_train = int(len(ls_data) * train_ratio)
    train_data = ls_data[:n_train]
    val_data = ls_data[n_train:]

    print(f"Разделение: {len(train_data)} train, {len(val_data)} val")

    def process_dataset(data: List[Dict], images_out: Path, labels_out: Path, split_name: str):
        """Обрабатывает подмножество данных (train/val)"""
        processed_count = 0

        for item in data:
            # Получаем имя файла
            if 'data' in item and 'image' in item['data']:
                image_url = item['data']['image']
                # Извлекаем имя файла из URL (после последнего слеша)
                image_filename = image_url.split('/')[-1].split('?')[0]  # убираем query параметры
            else:
                continue

            image_path = Path(images_dir) / image_filename
            if not image_path.exists():
                print(f"Предупреждение: изображение {image_filename} не найдено")
                continue

            # Копируем изображение
            shutil.copy2(image_path, images_out / image_filename)

            # Получаем размеры изображения
            original_width = None
            original_height = None

            # Ищем размеры в результатах аннотаций
            if 'annotations' in item:
                for annotation in item['annotations']:
                    if 'result' in annotation:
                        for result in annotation['result']:
                            if 'original_width' in result and 'original_height' in result:
                                original_width = result['original_width']
                                original_height = result['original_height']
                                break
                        if original_width and original_height:
                            break

            if not original_width or not original_height:
                print(f"Предупреждение: не найдены размеры для {image_filename}")
                continue

            # Создаем файл аннотаций
            label_filename = image_filename.rsplit('.', 1)[0] + '.txt'
            label_path = labels_out / label_filename

            with open(label_path, 'w') as f:
                if 'annotations' in item:
                    for annotation in item['annotations']:
                        if 'result' in annotation:
                            for result in annotation['result']:
                                if (result.get('type') == 'rectanglelabels' and
                                    'value' in result and 'rectanglelabels' in result['value']):

                                    # Координаты в процентах
                                    x_percent = result['value']['x']
                                    y_percent = result['value']['y']
                                    w_percent = result['value']['width']
                                    h_percent = result['value']['height']

                                    # Конвертируем в YOLO формат (нормализованные координаты)
                                    x_center = (x_percent + w_percent / 2) / 100.0
                                    y_center = (y_percent + h_percent / 2) / 100.0
                                    w_norm = w_percent / 100.0
                                    h_norm = h_percent / 100.0

                                    # Получаем класс (с нормализацией)
                                    label = result['value']['rectanglelabels'][0]
                                    normalized_label = normalize_class_name(label)
                                    class_id = class_to_id[normalized_label]

                                    # Записываем в файл
                                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

            processed_count += 1

        print(f"{split_name}: обработано {processed_count} изображений")

    # Обрабатываем train и val
    process_dataset(train_data, images_train, labels_train, "Train")
    process_dataset(val_data, images_val, labels_val, "Val")

    # Создаем data.yaml
    data_yaml = {
        'path': str(output_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': class_list
    }

    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

    print(f"Создан data.yaml: {yaml_path}")
    return str(yaml_path)

if __name__ == "__main__":
    # Пути к данным
    LS_JSON = "/home/ubuntu/diabert/dataset/predrazmetka_dashi/dinov3-sam2-gdino_20251001_140104/project-5-at-2025-10-02-11-31-782d7508.json"
    IMAGES_DIR = "/home/ubuntu/diabert/dataset/predrazmetka_dashi/dinov3-sam2-gdino_20251001_140104"
    OUTPUT_DIR = "/home/ubuntu/diabert/dataset/yolo_tools_dataset"

    # Конвертируем датасет
    data_yaml_path = convert_ls_to_yolo(
        ls_json_path=LS_JSON,
        images_dir=IMAGES_DIR,
        output_dir=OUTPUT_DIR,
        train_ratio=0.8,
        val_ratio=0.2
    )

    print("\nКонвертация завершена!")
    print(f"Датасет YOLO создан в: {OUTPUT_DIR}")
    print(f"Конфигурационный файл: {data_yaml_path}")
