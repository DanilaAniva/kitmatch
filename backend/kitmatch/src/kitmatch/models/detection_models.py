from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class DetectionRequest(BaseModel):
    image: Dict[str, str] = Field(..., description="Изображение в формате data URI")
    
class DetectionResponse(BaseModel):
    task_id: str
    image_id: str
    classes: List[str]
    scores: List[float]
    bboxes: List[Dict[str, Any]]
    detection_time: float
    overlay: str

