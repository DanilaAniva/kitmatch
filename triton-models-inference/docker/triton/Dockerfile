# Используем официальный образ Triton Inference Server
FROM nvcr.io/nvidia/tritonserver:23.09-py3

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл с зависимостями и локальный пакет 'models'
COPY requirements.txt .
COPY models ./models

# Устанавливаем переменные окружения
# Добавляем директорию 'models/src' в PYTHONPATH для корректной работы импортов
ENV PYTHONPATH="/app/models/src:${PYTHONPATH}"
ENV PIP_DEFAULT_TIMEOUT=180

# Устанавливаем системные зависимости и Python пакеты
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-opencv \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    # Устанавливаем зависимости, специфичные для модели YOLO
    && pip install -r ./models/src/models/YOLO/requirements.txt \
    # Устанавливаем локальный пакет 'models' в режиме редактирования
    && pip install --no-cache-dir -e ./models

# Открываем порты Triton
EXPOSE 8000 8001 8002

# Устанавливаем команду для запуска Triton сервера
# Репозиторий моделей монтируется через docker-compose, поэтому мы указываем на него здесь.
CMD ["tritonserver", "--model-repository=/model_repository", "--log-verbose=1", "--exit-on-error=false"]
