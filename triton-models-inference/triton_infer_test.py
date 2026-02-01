"""
Файл: /home/ubuntu/triton-models-inference/triton_infer_test.py

Основной смысл: Клиентский скрипт для тестирования YOLO модели детекции объектов 
через Triton Inference Server. Обеспечивает полный цикл: получение списка моделей,
выполнение инференса, сохранение результатов в JSON и визуализацию детекций.

Ключевые возможности:
- Получение списка доступных моделей с Triton сервера
- Выполнение инференса YOLO модели для детекции объектов
- Сохранение результатов в формате JSON
- Визуализация результатов детекции с bounding boxes и подписями
- Поддержка параметров через командную строку

Использование:
    python triton_infer_test.py --image path/to/image.jpg --host localhost --port 1339
"""

import os, sys, json, argparse
import http.client
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

# ===========================================
# НАСТРОЙКИ ТРИТОН СЕРВЕРА И МОДЕЛИ
# ===========================================

# Адрес Triton Inference Server
TRITON_HOST = "localhost"
TRITON_PORT = 1339

# Название модели для инференса
MODEL_NAME = "yolo11"

# ===========================================


def get_models(host: str, port: int) -> list:
    """
    Получает список доступных моделей с Triton Inference Server.
    
    Args:
        host (str): Адрес Triton сервера
        port (int): Порт Triton сервера
        
    Returns:
        list: Список словарей с информацией о доступных моделях
        
    Raises:
        RuntimeError: Если запрос к серверу завершился с ошибкой
        
    Example:
        >>> models = get_models("localhost", 1339)
        >>> print([m['name'] for m in models])
        ['yolo', 'resnet50']
    """
    conn = http.client.HTTPConnection(host, port)
    conn.request('POST', '/v2/repository/index', body='{}', headers={'Content-Type':'application/json'})
    resp = conn.getresponse(); data = resp.read().decode('utf-8')
    if resp.status != 200:
        raise RuntimeError(f"repo index failed: {resp.status} {resp.reason} {data}")
    return json.loads(data)


def infer_object_detector(host: str, port: int, model: str, image_path: str) -> dict:
    """
    Выполняет инференс модели детекции объектов на изображении.
    
    Args:
        host (str): Адрес Triton сервера
        port (int): Порт Triton сервера  
        model (str): Название модели для инференса
        image_path (str): Путь к изображению для анализа
        
    Returns:
        dict: Список детекций, каждая содержит bbox, confidence, class
        
    Raises:
        RuntimeError: Если инференс завершился с ошибкой
        FileNotFoundError: Если изображение не найдено
        
    Example:
        >>> results = infer_object_detector("localhost", 1339, "yolo", "test.jpg")
        >>> print(f"Найдено объектов: {len(results)}")
    """
    img = Image.open(image_path).convert('RGB')
    arr = np.array(img, dtype=np.uint8)
    raw = arr.tobytes(order='C')
    meta = {
        "inputs": [{
            "name": "image",
            "datatype": "UINT8",
            "shape": list(arr.shape),
            "parameters": {"binary_data_size": len(raw)}
        }],
        "outputs": [{"name": "bboxes"}]
    }
    body = json.dumps(meta).encode('utf-8')
    conn = http.client.HTTPConnection(host, port)
    conn.putrequest('POST', f'/v2/models/{model}/infer')
    conn.putheader('Content-Type', 'application/octet-stream')
    conn.putheader('Inference-Header-Content-Length', str(len(body)))
    conn.putheader('Content-Length', str(len(body) + len(raw)))
    conn.endheaders(); conn.send(body); conn.send(raw)
    resp = conn.getresponse(); data = resp.read().decode('utf-8')
    if resp.status != 200:
        raise RuntimeError(f"infer failed: {resp.status} {resp.reason} {data}")
    oj = json.loads(data)
    # outputs[0].data[0] contains JSON string
    out = oj['outputs'][0]['data'][0]
    return json.loads(out)


def save_json(out_obj: dict, path: str) -> None:
    """
    Сохраняет результаты инференса в JSON файл.
    
    Args:
        out_obj (dict): Результаты инференса для сохранения
        path (str): Путь для сохранения JSON файла
        
    Returns:
        None
        
    Side Effects:
        Создает директории при необходимости, перезаписывает существующий файл
        
    Example:
        >>> results = [{"bbox": [10, 20, 100, 200], "class": "screwdriver", "confidence": 0.95}]
        >>> save_json(results, "output/result.json")
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)


def save_vis(image_path: str, detections: list, vis_path: str) -> None:
    """
    Создает визуализацию результатов детекции с bounding boxes и подписями.
    
    Args:
        image_path (str): Путь к исходному изображению
        detections (list): Список детекций с bbox, class, confidence
        vis_path (str): Путь для сохранения визуализации
        
    Returns:
        None
        
    Side Effects:
        Создает директории при необходимости, сохраняет изображение в высоком качестве
        
    Features:
        - Автоматическое назначение цветов для разных классов
        - Подписи с названием класса и уверенностью
        - Легенда с перечнем всех классов
        - Высокое качество сохранения (300 DPI)
        
    Example:
        >>> detections = [{"bbox": [10, 20, 100, 200], "class": "hammer", "confidence": 0.87}]
        >>> save_vis("input.jpg", detections, "output/vis.jpg")
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import cv2

    # Загрузка изображения через OpenCV для лучшей совместимости
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка чтения изображения: {image_path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    # Цвета для разных классов
    class_names = [d.get('class', 'unknown') for d in detections]
    unique_classes = list(dict.fromkeys(class_names))  # сохраняем порядок
    colors = plt.cm.Set3(np.linspace(0, 1, max(len(unique_classes), 1)))
    class_to_color = {cls: colors[i % len(colors)] for i, cls in enumerate(unique_classes)}

    # Создание фигуры
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.imshow(image)

    # Отрисовка bounding boxes
    for d in detections:
        bbox = d.get('bbox', [])
        if len(bbox) >= 4:
            x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            cls_name = str(d.get('class', ''))
            score = d.get('confidence', None)

            # Выбор цвета по классу
            color = class_to_color.get(cls_name, (1.0, 0.0, 0.0, 1.0))

            # Рисуем прямоугольник
            rect = patches.Rectangle((x1, y1), w, h, linewidth=3,
                                    edgecolor=color, facecolor='none', alpha=0.9)
            ax.add_patch(rect)

            # Создаем лейбл
            label = cls_name if score is None else f"{cls_name} {score:.2f}"

            # Рисуем текст с красивым фоном
            ax.text(x1, max(0, y1 - 15), label, fontsize=14, color='white',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.9),
                    fontweight='bold', verticalalignment='bottom')

    # Настройки отображения
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis('off')
    plt.title(f'YOLO Detection Results', fontsize=16, fontweight='bold', pad=20)

    # Легенда по классам
    if unique_classes:
        legend_elements = [patches.Patch(color=class_to_color[c], label=c) for c in unique_classes]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=12)

    plt.tight_layout()

    # Сохранение с высоким качеством
    os.makedirs(os.path.dirname(vis_path), exist_ok=True)
    plt.savefig(vis_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Визуализация сохранена: {vis_path}")


def main():
    """
    Главная функция для запуска тестирования Triton Inference Server.
    
    Обрабатывает аргументы командной строки, выполняет полный цикл тестирования:
    1. Получение списка доступных моделей
    2. Проверка наличия целевой модели
    3. Выполнение инференса
    4. Сохранение результатов и визуализации
    
    Параметры командной строки:
        --host: Адрес Triton сервера (по умолчанию: localhost)
        --port: Порт Triton сервера (по умолчанию: 1339)
        --model: Название модели (по умолчанию: yolo)
        --image: Путь к изображению для тестирования
        --out_json: Путь для сохранения результатов в JSON
        --out_vis: Путь для сохранения визуализации
        
    Returns:
        None
        
    Example:
        $ python triton_infer_test.py --image test.jpg --host 192.168.1.100 --port 8000
    """
    ap = argparse.ArgumentParser(description='Triton Inference Test для YOLO модели детекции инструментов')

    # Используем настройки из переменных по умолчанию, но позволяем переопределить
    ap.add_argument('--host', default=TRITON_HOST,
                    help=f'Адрес Triton сервера (по умолчанию: {TRITON_HOST})')
    ap.add_argument('--port', type=int, default=TRITON_PORT,
                    help=f'Порт Triton сервера (по умолчанию: {TRITON_PORT})')
    ap.add_argument('--model', default=MODEL_NAME,
                    help=f'Название модели (по умолчанию: {MODEL_NAME})')
    ap.add_argument('--image', default='/home/ubuntu/diabert/dataset/predrazmetka_dashi/dinov3-sam2-gdino_20251001_140104/DSCN4946.JPG',
                    help='Путь к изображению для тестирования')
    ap.add_argument('--out_json', default='/home/ubuntu/triton_infer_out/result.json',
                    help='Путь для сохранения результатов в JSON')
    ap.add_argument('--out_vis', default='/home/ubuntu/triton_infer_out/result.jpg',
                    help='Путь для сохранения визуализации')

    args = ap.parse_args()

    print("🚀 Запуск тестирования Triton Inference Server")
    print(f"📍 Сервер: {args.host}:{args.port}")
    print(f"🤖 Модель: {args.model}")
    print(f"🖼️ Изображение: {args.image}")
    print("=" * 50)

    try:
        # Получаем список доступных моделей
        models = get_models(args.host, args.port)
        print('📋 Доступные модели:', models)

        # Проверяем, что наша модель есть в списке
        model_names = [m['name'] for m in models]
        if args.model not in model_names:
            print(f"❌ Ошибка: Модель '{args.model}' не найдена на сервере!")
            print(f"📋 Доступные модели: {model_names}")
            return

        # Выполняем инференс
        print(f"🔍 Выполняем инференс модели '{args.model}'...")
        res = infer_object_detector(args.host, args.port, args.model, args.image)

        # Выводим краткую информацию о результатах
        num_detections = len(res)
        if num_detections > 0:
            classes = [d.get('class', 'unknown') for d in res]
            unique_classes = list(set(classes))
            print(f"✅ Обнаружено {num_detections} объектов")
            print(f"🏷️ Классы: {unique_classes}")
        else:
            print("⚠️ Объекты не обнаружены")

        print('📊 Результаты инференса:', json.dumps(res, ensure_ascii=False)[:300] + '...' if len(json.dumps(res)) > 300 else json.dumps(res, ensure_ascii=False))

        # Сохраняем результаты
        save_json(res, args.out_json)
        save_vis(args.image, res, args.out_vis)
        print('💾 Сохранено:')
        print(f"   📄 JSON: {args.out_json}")
        print(f"   🖼️ Визуализация: {args.out_vis}")

        print("\n🎉 Тестирование завершено успешно!")

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == '__main__':
    main()


