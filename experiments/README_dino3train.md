### Обучение эмбеддера DINOv3 + ArcFace для инструментов

#### Назначение
Модель эмбеддингов обучается на 11 классах инструментов и используется для предразметки: по bbox/маске найденных объектов в проде присваивается класс на основе эмбеддингов и прототипов.

#### Структура данных
- Корень: `data_root/<class>/`
- Тренировочные кропы: `data_root/<class>/cropped/*.jpg`
- Опциональные маски: `data_root/<class>/masks/*_mask.png` или `*.png`
- Опциональные вырезы: `data_root/<class>/cutouts/*.png` (RGBA, альфа — маска)

#### Особенности тренировки
- Бэкенд: `timm` DINOv3 (`vit_large_patch16_dinov3.lvd1689m`) с глобальным пулом `token`.
- Голова: MLP до размера эмбеддинга `emb_dim`, L2-нормализация.
- Классификатор: ArcFace (аргументы `s`, `m` зашиты в коде).
- Аугментации: Albumentations (геометрия, цвет, шум, dropout, нормализация).
- Сплиты: стратифицированные `train/val/test`.
- Прототипы: после обучения для каждого класса считаются centroid и stack эмбеддингов; сохраняются в `prototypes.npz`.
- Логгирование:
  - JSONL: `out_dir/metrics.jsonl` (по эпохам, best, test; включает пер-класс метрики)
  - ClearML по флагу `--use_clearml` (макро и пер-класс графики train/val/test)
- Маски:
  - Режимы `--crop_mode`: `bbox`, `mask`, `mix` (в `mix` фон зануляется по маске с вероятностью `--mix_mask_p`).
  - Если включено `--auto_generate_masks` и масок нет, генерируем SAM-2 маски для каждого кропа.

#### Команда запуска (bbox, без масок)
```bash
python /home/ubuntu/diabert/dataset/predrazmetka_dashi/lct-dino-3.py \
  --data_root "/home/ubuntu/diabert/dataset/dataset" \
  --out_dir "/home/ubuntu/diabert/dataset/predrazmetka_dashi/lct_dino3_out" \
  --model_id "vit_large_patch16_dinov3.lvd1689m" \
  --emb_dim 128 \
  --freeze_backbone \
  --epochs 15 \
  --batch 64 \
  --lr 1e-3 \
  --img_size 224 \
  --workers 4 \
  --seed 42 \
  --val_split 0.2 \
  --test_split 0.1 \
  --save_every 5 \
  --min_per_class 200 \
  --crop_mode bbox \
  --use_clearml \
  --clearml_project "DINOv3-Embeddings" \
  --clearml_task_name "dino3_tools_11cls"
```

#### Команда запуска с масками (mix)
```bash
python /home/ubuntu/diabert/dataset/predrazmetka_dashi/lct-dino-3.py \
  --data_root "/home/ubuntu/diabert/dataset/dataset" \
  --out_dir "/home/ubuntu/diabert/dataset/predrazmetka_dashi/lct_dino3_out" \
  --model_id "vit_large_patch16_dinov3.lvd1689m" \
  --emb_dim 128 \
  --freeze_backbone \
  --epochs 15 \
  --batch 64 \
  --lr 1e-3 \
  --img_size 224 \
  --workers 4 \
  --seed 42 \
  --val_split 0.2 \
  --test_split 0.1 \
  --save_every 5 \
  --min_per_class 200 \
  --crop_mode mix \
  --mix_mask_p 0.4 \
  --auto_generate_masks \
  --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam2_ckpt /home/ubuntu/sam2/checkpoints/sam2.1_hiera_large.pt \
  --use_clearml \
  --clearml_project "DINOv3-Embeddings" \
  --clearml_task_name "dino3_tools_11cls_mix"
```

#### Выходные артефакты
- Чекпоинты: `checkpoint_last.pth`, `checkpoint_best.pth`, периодические `checkpoint_epXXX.pth`
- Логи: `train.log`, `metrics.jsonl`
- Прототипы: `prototypes.npz` (`centroids`, `stacks`, `cls2id`, `id2cls`)

#### Завершение
- При `KeyboardInterrupt` сохраняется `checkpoint_interrupt.pth`, метрики записываются, ClearML ран корректно закрывается.
- При штатном завершении рана — метрики и артефакты сохранены, ClearML ран закрыт, выводится сообщение об успехе.


