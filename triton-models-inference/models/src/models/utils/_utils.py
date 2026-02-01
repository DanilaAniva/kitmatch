"""
Файл: /home/ubuntu/triton-models-inference/models/src/models/utils/_utils.py

Основной смысл: Утилитарные функции для мониторинга и логирования ресурсов GPU.
Предоставляет функции для получения детальной информации об использовании видеопамяти
и производительности GPU в процессе инференса.

Ключевые возможности:
- Мониторинг использования GPU памяти через nvidia-smi
- Получение статистики использования памяти PyTorch
- Логирование информации о ресурсах с использованием кастомного логгера
- Обработка ошибок при недоступности GPU

Использование:
    from models.utils import log_gpu_usage
    
    # Логирование использования GPU
    log_gpu_usage("cuda:0", logger)
"""

import numpy as np
import cv2
import torch
import subprocess
import re


def log_gpu_usage(device, logger):
    """
    Логирует детальную информацию об использовании GPU памяти и ресурсов.
    
    Args:
        device (str): Устройство CUDA (например, "cuda:0")
        logger: Объект логгера с методом log_info для вывода информации
        
    Returns:
        None
        
    Side Effects:
        Выводит в лог информацию о:
        - Общем объеме GPU памяти
        - Использованной системной памяти (через nvidia-smi)
        - Зарезервированной памяти PyTorch
        - Выделенной памяти в процессе
        
    Обработка ошибок:
        - При недоступности GPU выводит сообщение об ошибке
        - При проблемах с nvidia-smi продолжает выполнение
        
    Example:
        >>> class SimpleLogger:
        ...     def log_info(self, msg):
        ...         print(f"[INFO] {msg}")
        >>> logger = SimpleLogger()
        >>> log_gpu_usage("cuda:0", logger)
        [INFO] Total GPU Memory: 8192.00 MiB
        [INFO] System Used Memory: 2048 MiB (25.00%) of 8192 MiB
    """
    try:
        if torch.cuda.is_available():
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,nounits,noheader'],
                            stdout=subprocess.PIPE)
            output = result.stdout.decode('utf-8').strip().split('\n')

            device = torch.device(device)
            total_memory = torch.cuda.get_device_properties(device).total_memory
            reserved_memory = torch.cuda.memory_reserved(device)
            allocated_memory = torch.cuda.memory_allocated(device)

            reserved_percentage = (reserved_memory / total_memory) * 100
            allocated_percentage = (allocated_memory / total_memory) * 100

            logger.log_info(f"Total GPU Memory: {total_memory / (1024 ** 2):.2f} MiB")
            for i, line in enumerate(output):
                total, used, free = map(int, re.findall(r'\d+', line))
                used_percentage = (used / total) * 100
                logger.log_info(f"System Used Memory: {used} MiB ({used_percentage:.2f}%) of {total} MiB")

            logger.log_info(f"Torch Reserved Memory: {reserved_memory / (1024 ** 2):.2f} MiB ({reserved_percentage:.2f}%)")
            logger.log_info(f"Torch Allocated Memory in Process: {allocated_memory / (1024 ** 2):.2f} MiB ({allocated_percentage:.2f}%)")
    except Exception:
        logger.log_info('log_gpu_usage error')