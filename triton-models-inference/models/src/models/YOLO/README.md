# YOLO Model for Triton Inference Server

This module provides YOLO (You Only Look Once) object detection capabilities for Triton Inference Server.

## Model Details

- **Architecture**: YOLOv8 nano (yolov8n)
- **Task**: Object detection
- **Classes**: 11 tool classes (Бокорезы, Ключ рожковый/накидной ¾, Коловорот, etc.)
- **Input**: RGB images
- **Output**: Bounding boxes with class labels and confidence scores

## Setup

1. Place the trained model weights in the `weights/` directory as `best.pt`
2. The model will be automatically loaded by the Triton Python backend

## Configuration

Model configuration is handled through the `config.pbtxt` file in the model repository.
