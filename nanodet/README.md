# NanoDet Object Detection with ONNX Runtime

This is a Python implementation of the NanoDet object detection model using ONNX Runtime for inference.

## Requirements

- Python 3.10 or higher
- OpenCV
- NumPy
- ONNX Runtime

## Installation

1. Create and activate a Python virtual environment (recommended):

```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate  # Windows
```

2. Install the required packages:

```bash
   pip install -r requirements.txt
```

## Usage

```bash
python detect.py --imgpath helmet_jacket_10256.jpg --modelpath helmet_jacket_detection.onnx --classfile class.names --conf-threshold 0.4 --nms-threshold 0.6
```