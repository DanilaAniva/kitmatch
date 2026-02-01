### Тренировка эмбеддера инструментов (ArcFace + DINOv3)

Файл: `lct-dino-3.py`

Назначение: обучает эмбеддер на базе `timm` DINOv3 (`vit_large_patch16_dinov3.lvd1689m`) с головой ArcFace. Сохраняет чекпоинты и `prototypes.npz` (центроиды и стеки эталонов) для дальнейшего косинусного мэтчинга.

Основные возможности:
- Аугментации Albumentations, нормализация по конфигу timm
- Чекпоинты: last/best/по эпохам, резюмирование `--resume`
- Логирование в `train.log` и `config.json` (включая параметры оптимизатора и лосса)
- Метрики на train/val: accuracy, macro-F1, macro ROC AUC
- DataParallel по флагу `--gpus N` (например, 2)

Минимальная команда запуска (пример):
```bash
python3 /home/ubuntu/diabert/dataset/predrazmetka_dashi/lct-dino-3.py \
  --data_root "/home/ubuntu/diabert/dataset/dataset" \
  --out_dir "/home/ubuntu/diabert/dataset/predrazmetka_dashi/lct_dino3_out" \
  --model_id "vit_large_patch16_dinov3.lvd1689m" \
  --epochs 15 --batch 64 --lr 1e-3 --img_size 224 \
  --workers 4 --seed 42 --val_split 0.2 --save_every 5 \
  --max_refs_per_class 16 \
  --gpus 2
```

Параметры:
- `--data_root`: корень датасета. Классы — подпапки. Если есть `cropped/`, берём картинки оттуда; иначе — из корня класса.
- `--out_dir`: папка для логов, чекпоинтов и `prototypes.npz`.
- `--model_id`: timm-модель DINOv3. По умолчанию `vit_large_patch16_dinov3.lvd1689m`.
- `--freeze_backbone`: заморозить бэкбон (по умолчанию обучается голова; можно выключить флаг — будет fine-tune).
- `--epochs`, `--batch`, `--lr`, `--img_size`, `--workers`, `--seed`, `--val_split`, `--save_every` — стандартные.
- `--resume`: путь к чекпоинту для продолжения (если не указан — автоиспользуется `checkpoint_last.pth`, если есть).
- `--max_refs_per_class`: лимит эталонов при построении прототипов (None — без лимита).
- `--gpus`: число GPU для DataParallel (например, 2).

Результаты:
- `train.log` — метрики по эпохам (train/val): loss, acc, F1 (macro), ROC AUC (macro)
- `config.json` — параметры запуска + сведения об оптимизаторе и лоссе
- `checkpoint_last.pth`, `checkpoint_best.pth`, `checkpoint_epXXX.pth` — веса
- `prototypes.npz` — центроиды/стеки эмбеддингов по классам

Зависимости:
- `torch`, `timm`, `albumentations`, `scikit-learn`, `tqdm`, `Pillow`


