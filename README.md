# AIsystem-2502

AI System - Face Recognition with Triton Inference Server (HW2)

## Overview

This project implements a face recognition system using:
- **Triton Inference Server** (NVIDIA) for model serving
- **FastAPI** for REST API wrapper
- **InsightFace** models (SCRFD detector, ArcFace recognition)
- **Docker** for easy deployment

All deep learning inference (face detection and recognition) runs on Triton, while Python handles preprocessing, postprocessing, and orchestration.

## Quick Start (이 파트로 test하시면 됩니다. 이 파트 밑은 프로젝트 디테일.)

```bash
# 0. Create virtual envs and clone

# ***Create virtual environment and activate(for extracting onnx files)***
conda create -n <NAME> python=3.10
conda activate <NAME>

# Clone into the desired folder
git clone https://github.com/wonseok289/AIsystem-2502.git

# Navigate to project directory
cd AIsystem-2502
cd DockerFRTriton

# 1. Extract models
pip install -r requirements.txt
python extract_insightface_models.py --output-dir model_repository

# 2. Build Docker
docker build -t fr-triton -f Docker/Dockerfile .

# 3. Run
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -p 3000:3000 --name fr_triton fr-triton

# 4. Test
Open browser: http://localhost:3000/docs
```

## Project Structure

```
AIsystem-2502/
├── README.md                          # This file
├── DockerFR/                          # HW1 (reference implementation)
├── DockerOnnxDetector/                # Detector reference (not used in final)
└── DockerFRTriton/                    # HW2 (main project)
    ├── model_repository/              # Triton model repository (generated)
    │   ├── face_detector/
    │   │   ├── 1/model.onnx
    │   │   └── config.pbtxt
    │   └── fr_model/
    │       ├── 1/model.onnx
    │       └── config.pbtxt
    ├── Docker/
    │   ├── Dockerfile
    │   └── start.sh
    ├── app.py                         # FastAPI application
    ├── pipeline.py                    # Face recognition pipeline
    ├── triton_service.py              # Triton server management
    ├── extract_insightface_models.py  # Model extraction script
    └── requirements.txt
```


## Setup: Extract InsightFace Models to ONNX

**IMPORTANT:** Before building Docker, you must extract the ONNX models.

```bash
cd DockerFRTriton

# Install dependencies (if not already installed)
pip install -r requirements.txt

# Extract InsightFace models (detector + FR) to model_repository
python extract_insightface_models.py --output-dir model_repository
```

This will:
- Download InsightFace models automatically on first run (~200MB)
- Copy detector ONNX to `model_repository/face_detector/1/model.onnx`
- Copy FR model ONNX to `model_repository/fr_model/1/model.onnx`
- Generate `model_info.txt` with model details

Verify models exist:
```bash
ls model_repository/face_detector/1/model.onnx
ls model_repository/fr_model/1/model.onnx
```

## Run with Docker + FastAPI (all-in-one container)

```bash
cd DockerFRTriton

# Build image (uses model_repository/*)
docker build -t fr-triton -f Docker/Dockerfile .

# Run container (Triton on 8000/8001/8002, FastAPI on 3000)
docker run --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -p 3000:3000 \
  --name fr_triton \
  fr-triton
```

Open Swagger UI at:
```
http://localhost:3000/docs
```


## Implementation Details

### Key Files

- **`extract_insightface_models.py`**: Extracts InsightFace detector and FR models to ONNX format and copies them to `model_repository/`.

- **`triton_service.py`**: 
  - Prepares `config.pbtxt` for both FR model and face detector
  - Starts/stops Triton server
  - Creates HTTP client with correct URL format
  - Provides inference helper functions

- **`pipeline.py`**: Full pipeline implementation
  - Calls Triton face detector for face detection and landmark extraction
  - Postprocesses detector outputs (anchor decoding, NMS, thresholding)
  - Scales landmarks from detector resolution (640x640) to original image size
  - Aligns faces using `cv2.warpAffine` with ArcFace standard coordinates
  - Calls Triton FR model for embedding extraction
  - Computes cosine similarity between embeddings

- **`app.py`**: FastAPI wrapper exposing `/embedding`, `/face-similarity`, `/health`. Keeps business logic in `pipeline.py` with all model inference on Triton.(Original file, not modified from template)

### Architecture

```
User Request (JPEG/PNG images)
    ↓
FastAPI (app.py)
    ↓
Pipeline (pipeline.py)
    ├→ Triton Detector (SCRFD) → Raw outputs (scores, bboxes, landmarks)
    ├→ Python Postprocessing (decode anchors, NMS, scale to original size)
    ├→ Face Alignment (cv2.warpAffine with ArcFace landmarks)
    ├→ Triton FR Model (ArcFace) → Embeddings
    └→ Cosine Similarity Calculation
    ↓
Response (similarity score)
```

### Triton Model Repository

All models are served by Triton Inference Server:

- **face_detector** (SCRFD): Detects faces and facial landmarks
  - Input: `[1, 3, 640, 640]` RGB image (normalized)
  - Outputs: 9 tensors (3 scales × 3 types: scores, bboxes, landmarks)
  
- **fr_model** (ArcFace): Extracts face embeddings
  - Input: `[1, 3, 112, 112]` aligned face (normalized)
  - Output: `[1, 512]` embedding vector

Config files (`config.pbtxt`) are automatically generated at startup.

## Notes

- All models (FR, detector) are served by Triton; FastAPI does not run heavy model code locally.
- Supports JPEG and PNG images (including RGBA with transparency).
- Config files (`config.pbtxt`) are automatically generated based on ONNX model inspection.
- Anti-spoofing is not implemented (not required for HW2).


## Troubleshooting

### Memory Issues
Configure Docker Desktop with enough RAM (Settings → Resources → Memory).