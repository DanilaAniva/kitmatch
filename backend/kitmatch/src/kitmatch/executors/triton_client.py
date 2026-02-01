import http.client
import json
from typing import Dict, Any
import numpy as np
from PIL import Image

def infer_yolo_object_detector(
    host: str,
    port: int,
    model_name: str,
    image: np.ndarray,
) -> Dict[str, Any]:
    """
    Выполняет инференс модели YOLO на сервере Triton с использованием нативного HTTP клиента.

    Args:
        host (str): Адрес Triton Inference Server.
        port (int): Порт Triton Inference Server.
        model_name (str): Название модели для инференса.
        image (np.ndarray): Изображение в формате NumPy array (H, W, C), dtype=uint8.

    Returns:
        Dict[str, Any]: Словарь с результатами детекции.
    
    Raises:
        RuntimeError: Если инференс завершился с ошибкой.
    """
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    raw = image.tobytes(order='C')
    meta = {
        "inputs": [{
            "name": "image",
            "datatype": "UINT8",
            "shape": list(image.shape),
            "parameters": {"binary_data_size": len(raw)}
        }],
        "outputs": [{"name": "bboxes"}]
    }
    body = json.dumps(meta).encode('utf-8')
    
    conn = http.client.HTTPConnection(host, str(port))
    try:
        conn.putrequest('POST', f'/v2/models/{model_name}/infer')
        conn.putheader('Content-Type', 'application/octet-stream')
        conn.putheader('Inference-Header-Content-Length', str(len(body)))
        conn.putheader('Content-Length', str(len(body) + len(raw)))
        conn.endheaders()
        conn.send(body)
        conn.send(raw)
        
        resp = conn.getresponse()
        data = resp.read().decode('utf-8')
        
        if resp.status != 200:
            raise RuntimeError(f"Infer failed: {resp.status} {resp.reason} | {data}")
        
        oj = json.loads(data)
        # В ответе Triton результат может быть вложен как строка JSON
        out = oj['outputs'][0]['data'][0]
        return json.loads(out)
    finally:
        conn.close()
