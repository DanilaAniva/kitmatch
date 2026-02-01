# Создание кропов для обучения

## Проверка текущего количества кропов

Для проверки количества исходных изображений и уже созданных кропов в каждом классе:

```bash
DATA_ROOT='/home/ubuntu/diabert/dataset/dataset'
echo "Проверка количества файлов по классам:"
echo "========================================"
for class_dir in "$DATA_ROOT"/*/; do
  class_name=$(basename "$class_dir")
  # Подсчет исходных изображений (в корне класса)
  orig_count=$(find "$class_dir" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.webp' \) | wc -l)
  # Подсчет кропов (в подпапке cropped)
  cropped_dir="${class_dir%/}/cropped"
  if [ -d "$cropped_dir" ]; then
    crop_count=$(find "$cropped_dir" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.webp' \) | wc -l)
  else
    crop_count=0
  fi
  printf "%-40s | исходных: %3d | кропов: %3d | нужно: %3d\n" "$class_name" "$orig_count" "$crop_count" "$((300 - crop_count > 0 ? 300 - crop_count : 0))"
done
```

## Создание дополнительных кропов для недостающих классов

Команды для создания кропов до 300 штук в каждом классе с использованием модели `rziga/mm_grounding_dino_large_all`:

### Для конкретных недостающих классов:

```bash
# Отвертка «-» (247 → 300)
python3 /home/ubuntu/diabert/dataset/predrazmetka_dashi/make_bbox_crops.py --class_dir '/home/ubuntu/diabert/dataset/dataset/1 Отвертка «-»' --max_n 300 --model_name 'rziga/mm_grounding_dino_large_all'

# Бокорезы (271 → 300) 
python3 /home/ubuntu/diabert/dataset/predrazmetka_dashi/make_bbox_crops.py --class_dir '/home/ubuntu/diabert/dataset/dataset/11 Бокорезы' --max_n 300 --model_name 'rziga/mm_grounding_dino_large_all'

# Отвертка на смещенный крест (275 → 300)
python3 /home/ubuntu/diabert/dataset/predrazmetka_dashi/make_bbox_crops.py --class_dir '/home/ubuntu/diabert/dataset/dataset/3 Отвертка на смещенный крест' --max_n 300 --model_name 'rziga/mm_grounding_dino_large_all'

# Открывашка для банок с маслом (101 → 300)
python3 /home/ubuntu/diabert/dataset/predrazmetka_dashi/make_bbox_crops.py --class_dir '/home/ubuntu/diabert/dataset/dataset/9 Открывашка для банок с маслом' --max_n 300 --model_name 'rziga/mm_grounding_dino_large_all'

# Разводной ключ (340 → 300, уже достаточно, но можно добавить)
python3 /home/ubuntu/diabert/dataset/predrazmetka_dashi/make_bbox_crops.py --class_dir '/home/ubuntu/diabert/dataset/dataset/8 Разводной ключ' --max_n 300 --model_name 'rziga/mm_grounding_dino_large_all'
```

### Автоматически для всех классов (добить до 300):

```bash
DATA_ROOT='/home/ubuntu/diabert/dataset/dataset'
MODEL='rziga/mm_grounding_dino_large_all'
for class_dir in "$DATA_ROOT"/*/; do
  class_name=$(basename "$class_dir")
  cropped_dir="${class_dir%/}/cropped"
  if [ -d "$cropped_dir" ]; then
    count=$(find "$cropped_dir" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.webp' \) | wc -l)
    if [ "$count" -lt 300 ]; then
      echo "Создаю кропы для: $class_name (текущее количество: $count)"
      python3 /home/ubuntu/diabert/dataset/predrazmetka_dashi/make_bbox_crops.py --class_dir "$class_dir" --max_n 300 --model_name "$MODEL" --threshold 0.10 --expand 0.06
    fi
  fi
done
```

## Параметры make_bbox_crops.py

- `--max_n` - максимальное количество кропов для создания (скрипт пропускает уже существующие)
- `--model_name` - модель GroundingDINO для детекции
- `--threshold` - порог детекции (по умолчанию 0.1)
- `--expand` - расширение bbox на каждую сторону (по умолчанию 0.06)

## Примечания

- Скрипт автоматически пропускает уже существующие кропы
- Если исходных изображений меньше чем `max_n`, будет создано столько кропов, сколько есть исходников
- Кропы автоматически поворачиваются в вертикальную ориентацию если нужно
