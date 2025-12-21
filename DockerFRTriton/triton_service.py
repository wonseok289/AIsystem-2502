import subprocess
import textwrap
import time
from pathlib import Path
from typing import Any

import numpy as np
import cv2


TRITON_HTTP_PORT = 8000
TRITON_GRPC_PORT = 8001
TRITON_METRICS_PORT = 8002

# FR Model configuration
MODEL_NAME = "fr_model"
MODEL_VERSION = "1"
MODEL_INPUT_NAME = "input.1"  # ONNX 검사 결과
MODEL_OUTPUT_NAME = "683"  # ONNX 검사 결과
MODEL_IMAGE_SIZE = (112, 112)

# Face Detector configuration
DETECTOR_MODEL_NAME = "face_detector"
DETECTOR_VERSION = "1"
DETECTOR_INPUT_NAME = "input.1"  # ONNX 검사 결과

# Detector outputs: 3 scales (large, medium, small) × 3 types (score, bbox, landmarks)
DETECTOR_OUTPUT_SCORES = ["448", "471", "494"]  # [12800,1], [3200,1], [800,1]
DETECTOR_OUTPUT_BBOXES = ["451", "474", "497"]  # [12800,4], [3200,4], [800,4]
DETECTOR_OUTPUT_LANDMARKS = ["454", "477", "500"]  # [12800,10], [3200,10], [800,10]
DETECTOR_IMAGE_SIZE = (640, 640)


def prepare_model_repository(model_repo: Path) -> None:
    """
    Populate the Triton model repository with ONNX models and config.pbtxt files.
    Prepares both FR model and face detector.
    """
    # Prepare FR model
    _prepare_fr_model(model_repo)
    
    # Prepare Face Detector
    _prepare_detector_model(model_repo)
    
    print(f"[triton] Model repository preparation complete: {model_repo}")


def _prepare_fr_model(model_repo: Path) -> None:
    """Prepare FR recognition model config."""
    model_dir = model_repo / MODEL_NAME / MODEL_VERSION
    model_path = model_dir / "model.onnx"
    config_path = model_dir.parent / "config.pbtxt"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing FR ONNX model at {model_path}. "
            "Run extract_insightface_models.py first."
        )

    model_dir.mkdir(parents=True, exist_ok=True)
    config_text = textwrap.dedent(
        f"""
        name: "{MODEL_NAME}"
        platform: "onnxruntime_onnx"
        max_batch_size: 0
        default_model_filename: "model.onnx"
        input [
          {{
            name: "{MODEL_INPUT_NAME}"
            data_type: TYPE_FP32
            dims: [1, 3, {MODEL_IMAGE_SIZE[0]}, {MODEL_IMAGE_SIZE[1]}]
          }}
        ]
        output [
          {{
            name: "{MODEL_OUTPUT_NAME}"
            data_type: TYPE_FP32
            dims: [1, 512]
          }}
        ]
        instance_group [
          {{ kind: KIND_CPU }}
        ]
        """
    ).strip() + "\n"

    config_path.write_text(config_text)
    print(f"[triton] Prepared FR model config: {config_path}")


def _prepare_detector_model(model_repo: Path) -> None:
    """Prepare face detector model config."""
    detector_dir = model_repo / DETECTOR_MODEL_NAME / DETECTOR_VERSION
    detector_path = detector_dir / "model.onnx"
    config_path = detector_dir.parent / "config.pbtxt"

    if not detector_path.exists():
        raise FileNotFoundError(
            f"Missing detector ONNX model at {detector_path}. "
            "Run extract_insightface_models.py first."
        )

    detector_dir.mkdir(parents=True, exist_ok=True)
    
    # Detector outputs 9 tensors: 3 scales × (score, bbox, landmarks)
    # Scores: [12800,1], [3200,1], [800,1]
    # BBoxes: [12800,4], [3200,4], [800,4]
    # Landmarks: [12800,10], [3200,10], [800,10]
    
    output_configs = []
    
    # Add score outputs
    output_configs.append(f'{{ name: "{DETECTOR_OUTPUT_SCORES[0]}" data_type: TYPE_FP32 dims: [12800, 1] }}')
    output_configs.append(f'{{ name: "{DETECTOR_OUTPUT_SCORES[1]}" data_type: TYPE_FP32 dims: [3200, 1] }}')
    output_configs.append(f'{{ name: "{DETECTOR_OUTPUT_SCORES[2]}" data_type: TYPE_FP32 dims: [800, 1] }}')
    
    # Add bbox outputs
    output_configs.append(f'{{ name: "{DETECTOR_OUTPUT_BBOXES[0]}" data_type: TYPE_FP32 dims: [12800, 4] }}')
    output_configs.append(f'{{ name: "{DETECTOR_OUTPUT_BBOXES[1]}" data_type: TYPE_FP32 dims: [3200, 4] }}')
    output_configs.append(f'{{ name: "{DETECTOR_OUTPUT_BBOXES[2]}" data_type: TYPE_FP32 dims: [800, 4] }}')
    
    # Add landmark outputs
    output_configs.append(f'{{ name: "{DETECTOR_OUTPUT_LANDMARKS[0]}" data_type: TYPE_FP32 dims: [12800, 10] }}')
    output_configs.append(f'{{ name: "{DETECTOR_OUTPUT_LANDMARKS[1]}" data_type: TYPE_FP32 dims: [3200, 10] }}')
    output_configs.append(f'{{ name: "{DETECTOR_OUTPUT_LANDMARKS[2]}" data_type: TYPE_FP32 dims: [800, 10] }}')
    
    output_section = "\n  ".join(output_configs)
    
    config_text = textwrap.dedent(
        f"""
        name: "{DETECTOR_MODEL_NAME}"
        platform: "onnxruntime_onnx"
        max_batch_size: 0
        default_model_filename: "model.onnx"
        input [
          {{
            name: "{DETECTOR_INPUT_NAME}"
            data_type: TYPE_FP32
            dims: [1, 3, {DETECTOR_IMAGE_SIZE[0]}, {DETECTOR_IMAGE_SIZE[1]}]
          }}
        ]
        output [
          {output_section}
        ]
        instance_group [
          {{ kind: KIND_CPU }}
        ]
        """
    ).strip() + "\n"

    config_path.write_text(config_text)
    print(f"[triton] Prepared detector model config: {config_path}")


def start_triton_server(model_repo: Path) -> Any:
    """
    Launch Triton Inference Server (CPU) pointing at model_repo and return a handle/process.
    """
    triton_bin = subprocess.run(["which", "tritonserver"], capture_output=True, text=True).stdout.strip()
    if not triton_bin:
        raise RuntimeError("Could not find `tritonserver` binary in PATH. Is Triton installed?")

    cmd = [
        triton_bin,
        f"--model-repository={model_repo}",
        f"--http-port={TRITON_HTTP_PORT}",
        f"--grpc-port={TRITON_GRPC_PORT}",
        f"--metrics-port={TRITON_METRICS_PORT}",
        "--allow-http=true",
        "--allow-grpc=true",
        "--allow-metrics=true",
        "--log-verbose=1",
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(f"[triton] Starting Triton server with command: {' '.join(cmd)}")
    time.sleep(3)  # Give the server a moment to load the model
    return process


def stop_triton_server(server_handle: Any) -> None:
    """
    Cleanly stop the Triton server started in start_triton_server.
    """
    if server_handle is None:
        return

    server_handle.terminate()
    try:
        server_handle.wait(timeout=10)
        print("[triton] Triton server stopped.")
    except subprocess.TimeoutExpired:
        server_handle.kill()
        print("[triton] Triton serverg killed after timeout.")


def create_triton_client(url: str) -> Any:
    """
    Initialize a Triton HTTP client for the FR model endpoint.
    
    Note: tritonclient expects URL without scheme (e.g., "localhost:8000" not "http://localhost:8000")
    """
    try:
        from tritonclient import http as httpclient
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError("tritonclient[http] is required; install from requirements.txt") from exc

    # Remove http:// or https:// scheme if present
    url_without_scheme = url.replace("http://", "").replace("https://", "")
    
    client = httpclient.InferenceServerClient(url=url_without_scheme, verbose=False)
    if not client.is_server_live():
        raise RuntimeError(f"Triton server at {url_without_scheme} is not live.")
    return client


def run_inference(client: Any, image_input: Any) -> Any:
    """
    Preprocess an input image for FR model, call Triton, and return embeddings.
    
    Args:
        client: Triton HTTP client
        image_input: Either bytes or numpy array (BGR, HWC, 112x112)
    
    Returns:
        Embedding vector of shape [1, 512].
    """
    try:
        from io import BytesIO
        from PIL import Image
        from tritonclient import http as httpclient
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError("Pillow, numpy, and tritonclient[http] are required to run inference.") from exc

    # Handle both bytes and numpy array inputs
    if isinstance(image_input, bytes):
        # Legacy path: decode from bytes
        with Image.open(BytesIO(image_input)) as img:
            img = img.convert("RGB").resize(MODEL_IMAGE_SIZE)
            np_img = np.asarray(img, dtype=np.float32)
    elif isinstance(image_input, np.ndarray):
        # Direct numpy array (BGR format from cv2)
        # Convert BGR to RGB
        np_img = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB).astype(np.float32)
    else:
        raise ValueError(f"Unsupported image_input type: {type(image_input)}")
    
    # InsightFace FR preprocessing: normalize to [-1, 1]
    # mean=127.5, std=127.5 → (pixel - 127.5) / 127.5
    np_img = (np_img - 127.5) / 127.5

    np_img = np.transpose(np_img, (2, 0, 1))  # HWC -> CHW
    batch = np.expand_dims(np_img, axis=0)  # Shape: [1, 3, 112, 112]

    infer_input = httpclient.InferInput(MODEL_INPUT_NAME, batch.shape, "FP32")
    infer_input.set_data_from_numpy(batch)

    infer_output = httpclient.InferRequestedOutput(MODEL_OUTPUT_NAME)
    response = client.infer(model_name=MODEL_NAME, inputs=[infer_input], outputs=[infer_output])
    return response.as_numpy(MODEL_OUTPUT_NAME)


def run_detector_inference(client: Any, image_bytes: bytes) -> dict:
    """
    Preprocess an input image for face detector, call Triton, and return raw detection outputs.
    
    Returns dict with:
        - 'scores': list of 3 score arrays (large, medium, small scale)
        - 'bboxes': list of 3 bbox arrays
        - 'landmarks': list of 3 landmark arrays
        - 'original_size': tuple (width, height) of original image
        - 'input_size': tuple (width, height) of detector input
        
    Postprocessing (NMS, thresholding, etc.) should be done by the caller in pipeline.py
    """
    try:
        from io import BytesIO
        from PIL import Image
        from tritonclient import http as httpclient
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError("Pillow, numpy, and tritonclient[http] are required to run inference.") from exc

    with Image.open(BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        original_size = img.size  # (width, height)
        img = img.resize(DETECTOR_IMAGE_SIZE)
        
        # InsightFace detector preprocessing: normalize to [-1, 1]
        # mean=127.5, std=128.0 → (pixel - 127.5) / 128.0
        np_img = np.asarray(img, dtype=np.float32)
        np_img = (np_img - 127.5) / 128.0

    np_img = np.transpose(np_img, (2, 0, 1))  # HWC -> CHW
    batch = np.expand_dims(np_img, axis=0)  # Shape: [1, 3, 640, 640]

    infer_input = httpclient.InferInput(DETECTOR_INPUT_NAME, batch.shape, "FP32")
    infer_input.set_data_from_numpy(batch)

    # Request all 9 outputs
    output_requests = []
    for score_name in DETECTOR_OUTPUT_SCORES:
        output_requests.append(httpclient.InferRequestedOutput(score_name))
    for bbox_name in DETECTOR_OUTPUT_BBOXES:
        output_requests.append(httpclient.InferRequestedOutput(bbox_name))
    for landmark_name in DETECTOR_OUTPUT_LANDMARKS:
        output_requests.append(httpclient.InferRequestedOutput(landmark_name))
    
    response = client.infer(
        model_name=DETECTOR_MODEL_NAME, 
        inputs=[infer_input], 
        outputs=output_requests
    )
    
    # Extract all outputs
    scores = [response.as_numpy(name) for name in DETECTOR_OUTPUT_SCORES]
    bboxes = [response.as_numpy(name) for name in DETECTOR_OUTPUT_BBOXES]
    landmarks = [response.as_numpy(name) for name in DETECTOR_OUTPUT_LANDMARKS]
    
    return {
        'scores': scores,
        'bboxes': bboxes,
        'landmarks': landmarks,
        'original_size': original_size,
        'input_size': DETECTOR_IMAGE_SIZE
    }
