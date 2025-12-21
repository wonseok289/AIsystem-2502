import io
from typing import Any, Tuple, List

import cv2
import numpy as np
from PIL import Image

from triton_service import run_detector_inference, run_inference


# ArcFace standard alignment target coordinates (112x112)
ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)


def _bytes_to_image(image_bytes: bytes) -> np.ndarray:
    """Convert image bytes to BGR numpy array. Handles RGB, RGBA, and grayscale."""
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB first (handles RGBA, L, etc.)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image_array = np.array(image)
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    return image_bgr


def _image_to_bytes(image_array: np.ndarray) -> bytes:
    """Convert BGR numpy array to JPEG bytes."""
    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    buffer = io.BytesIO()
    image_pil.save(buffer, format="JPEG")
    return buffer.getvalue()


def _distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box (for SCRFD-style detectors)."""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to landmarks (for SCRFD-style detectors)."""
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, 0] + distance[:, i]
        py = points[:, 1] + distance[:, i + 1]
        if max_shape is not None:
            px = np.clip(px, 0, max_shape[1])
            py = np.clip(py, 0, max_shape[0])
        preds.append(px)
        preds.append(py)
    # Stack along axis 1 to get shape [N, 10]
    return np.stack(preds, axis=1)


def nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors."""
    a_norm = np.linalg.norm(vec_a)
    b_norm = np.linalg.norm(vec_b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (a_norm * b_norm))


def postprocess_detections_scrfd(detector_output: dict, input_size=(640, 640), 
                                 score_threshold=0.5, nms_threshold=0.4) -> List[dict]:
    """
    Postprocess SCRFD-style detector outputs from Triton.
    
    SCRFD is anchor-free, outputs are relative to feature map grid points.
    """
    try:
        scores_list = detector_output['scores']
        bboxes_list = detector_output['bboxes']
        landmarks_list = detector_output['landmarks']
    except Exception as e:
        print(f"[ERROR] Failed to extract detector outputs: {e}")
        raise
    
    # Feature map strides for 3 scales
    fmc = 3  # number of feature map scales
    feat_stride_fpn = [8, 16, 32]
    
    # Calculate feature map sizes
    feature_map_sizes = []
    for stride in feat_stride_fpn:
        feature_map_sizes.append([input_size[0] // stride, input_size[1] // stride])
    
    # Generate anchor centers for each scale
    # InsightFace detector uses 2 anchors per location
    num_anchors = 2
    anchor_centers = []
    for idx, stride in enumerate(feat_stride_fpn):
        h, w = feature_map_sizes[idx]
        anchor_center = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32)
        anchor_center = (anchor_center * stride).reshape(-1, 2)
        # Repeat for num_anchors (2 anchors per location)
        anchor_center = np.repeat(anchor_center, num_anchors, axis=0)
        anchor_centers.append(anchor_center)
    
    # Collect all detections
    det_bboxes = []
    det_scores = []
    det_kps = []
    
    for idx in range(fmc):
        scores = scores_list[idx].flatten()
        bbox_preds = bboxes_list[idx]  # [N, 4]
        kps_preds = landmarks_list[idx]  # [N, 10]
        centers = anchor_centers[idx]
        
        # Filter by score
        valid_idx = np.where(scores >= score_threshold)[0]
        
        if len(valid_idx) == 0:
            continue
            
        scores = scores[valid_idx]
        bbox_preds = bbox_preds[valid_idx]
        kps_preds = kps_preds[valid_idx]
        centers = centers[valid_idx]
        
        # Decode bboxes (distance to bbox format)
        bbox_preds = bbox_preds * feat_stride_fpn[idx]
        bboxes = _distance2bbox(centers, bbox_preds)
        
        # Decode landmarks
        kps_preds = kps_preds * feat_stride_fpn[idx]
        kps = _distance2kps(centers, kps_preds)
        kps = kps.reshape(-1, 5, 2)
        
        det_bboxes.append(bboxes)
        det_scores.append(scores)
        det_kps.append(kps)
    
    if len(det_bboxes) == 0:
        return []
    
    # Concatenate all scales
    det_bboxes = np.vstack(det_bboxes)
    det_scores = np.hstack(det_scores)
    det_kps = np.vstack(det_kps)
    
    # Prepare for NMS
    dets = np.hstack([det_bboxes, det_scores[:, np.newaxis]])
    keep_indices = nms(dets, nms_threshold)
    
    # Build result
    results = []
    for idx in keep_indices:
        results.append({
            'bbox': det_bboxes[idx],
            'landmarks': det_kps[idx],
            'score': float(det_scores[idx])
        })
    
    return results


def align_face(image_bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """
    Align face using 5-point landmarks to ArcFace standard coordinates.
    
    Args:
        image_bgr: Input image in BGR format
        landmarks: 5 facial keypoints [5, 2] (left_eye, right_eye, nose, mouth_left, mouth_right)
    
    Returns:
        Aligned face image (112x112) in BGR format
    """
    result = cv2.estimateAffinePartial2D(
        landmarks,
        ARCFACE_DST,
        method=cv2.LMEDS
    )
    
    if result is None or result[0] is None:
        raise ValueError("Failed to estimate transformation matrix for face alignment")
    
    transformation_matrix = result[0]
    
    aligned_face = cv2.warpAffine(
        image_bgr,
        transformation_matrix,
        (112, 112),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    return aligned_face


def get_embeddings(client: Any, image_a: bytes, image_b: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full pipeline using Triton for both detection and FR.
    
    Pipeline:
    1. Triton detector → raw outputs
    2. Python postprocessing (decode anchors, NMS)
    3. Face alignment (cv2.warpAffine)
    4. Triton FR model → embeddings
    
    Args:
        client: Triton HTTP client
        image_a: First image as bytes
        image_b: Second image as bytes
    
    Returns:
        Tuple of two embedding vectors
    """
    # Convert bytes to images
    img_a = _bytes_to_image(image_a)
    img_b = _bytes_to_image(image_b)
    
    # Detect faces using Triton detector
    detector_output_a = run_detector_inference(client, image_a)
    faces_a = postprocess_detections_scrfd(detector_output_a)
    
    detector_output_b = run_detector_inference(client, image_b)
    faces_b = postprocess_detections_scrfd(detector_output_b)
    
    if len(faces_a) == 0:
        raise ValueError("No face detected in image A")
    if len(faces_b) == 0:
        raise ValueError("No face detected in image B")
    
    # Get the best face (highest score)
    face_a = faces_a[0]  # Already sorted by score
    face_b = faces_b[0]
    
    # Extract landmarks (5 points) - these are in 640x640 coordinate system
    landmarks_a = face_a['landmarks'].copy()  # shape: (5, 2)
    landmarks_b = face_b['landmarks'].copy()
    
    # Scale landmarks from detector input size (640x640) to original image size
    # Detector resizes images to 640x640, so we need to scale back
    detector_size = 640  # Fixed detector input size
    scale_a_x = img_a.shape[1] / detector_size
    scale_a_y = img_a.shape[0] / detector_size
    scale_b_x = img_b.shape[1] / detector_size
    scale_b_y = img_b.shape[0] / detector_size
    
    landmarks_a[:, 0] *= scale_a_x
    landmarks_a[:, 1] *= scale_a_y
    landmarks_b[:, 0] *= scale_b_x
    landmarks_b[:, 1] *= scale_b_y
    
    # Align faces using landmarks
    aligned_a = align_face(img_a, landmarks_a)
    aligned_b = align_face(img_b, landmarks_b)
    
    # Extract embeddings using Triton FR model
    # Pass numpy arrays directly (no JPEG compression)
    emb_a = run_inference(client, aligned_a)
    emb_b = run_inference(client, aligned_b)
    
    return emb_a.squeeze(), emb_b.squeeze()


def calculate_face_similarity(client: Any, image_a: bytes, image_b: bytes) -> float:
    """
    End-to-end face similarity pipeline using Triton for all model inference.
    
    Pipeline:
        1. Detect faces using Triton face_detector
        2. Postprocess detections (NMS, thresholding)
        3. Align faces using landmarks (cv2.warpAffine)
        4. Extract embeddings using Triton fr_model
        5. Calculate cosine similarity
    
    Args:
        client: Triton HTTP client
        image_a: First image as bytes
        image_b: Second image as bytes
    
    Returns:
        Similarity score between 0 and 1
    """
    emb_a, emb_b = get_embeddings(client, image_a, image_b)
    similarity = _cosine_similarity(emb_a, emb_b)
    
    # Normalize from [-1, 1] to [0, 1]
    similarity_normalized = (similarity + 1.0) / 2.0
    
    return float(similarity_normalized)
