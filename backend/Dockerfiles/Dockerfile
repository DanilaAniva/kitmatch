# Используем официальный образ Python
FROM python:3.11-slim

# --- Build-time Arguments ---
ARG API_HOST=0.0.0.0
ARG API_PORT_INTERNAL=8000

# Устанавливаем системные зависимости для OpenCV и uv
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/* && \
    curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Устанавливаем системные зависимости, включая шрифты с поддержкой кириллицы
RUN apt-get update && \
    apt-get install -y --no-install-recommends fonts-dejavu-core && \
    rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы зависимостей
COPY pyproject.toml README.md uv.lock ./

# Создаем venv и устанавливаем зависимости без локального пакета
RUN uv sync --frozen --no-dev --no-install-project

# Копируем исходный код после установки зависимостей
COPY kitmatch/ ./kitmatch/

# Добавляем .venv в PATH
ENV PATH="/app/.venv/bin:${PATH}"

# Настраиваем PYTHONPATH для правильного импорта
ENV PYTHONPATH="/app/kitmatch/src"

# Открываем порт
EXPOSE 8000

# Запуск приложения
CMD ["uvicorn", "kitmatch.main:app", "--host", "0.0.0.0", "--port", "8000"]
