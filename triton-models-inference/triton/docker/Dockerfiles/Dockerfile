FROM nvcr.io/nvidia/tritonserver:23.09-py3

COPY ../triton/inference/__init__.py /app/triton/inference/__init__.py
COPY ../models/setup.py /app/models/setup.py
COPY ../models/src/models/__init__.py /app/models/src/models/__init__.py
COPY ../requirements.txt /app/requirements.txt

WORKDIR /app/triton


ENV PIP_DEFAULT_TIMEOUT=180
RUN apt-get update && apt-get install -y python3-opencv git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r ../requirements.txt \
    && pip install --no-cache-dir -e ../models


CMD ["tail", "-F" , "anything"]
CMD ["tritonserver", "--model-repository=/app/triton/inference/model_repository", "--log-verbose=2", "--exit-on-error=false"]
