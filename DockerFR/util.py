"""
Utility stubs for the face recognition project.

Each function is intentionally left unimplemented so that students can
fill in the logic as part of the coursework.
"""
   
import io
import cv2
import numpy as np
from typing import Any, List
from PIL import Image
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity


ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)


def _bytes_to_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes))
    image_array = np.array(image)
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
    return image_bgr


def _get_face_app():
    global _face_app
    if _face_app is None:
        _face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
        _face_app.prepare(ctx_id=0, det_size=(640, 640))
    return _face_app


def detect_faces(image: Any) -> List[Any]:
    """
    Detect faces within the provided image.

    Parameters can be raw image bytes or a decoded image object, depending on
    the student implementation. Expected to return a list of face regions
    (e.g., bounding boxes or cropped images).
    """
    if isinstance(image, bytes):
        image = _bytes_to_image(image)
    
    app = _get_face_app()
    faces = app.get(image)
    
    if not faces:
        return []
    
    face_list = []
    for face in faces:
        bbox = face.bbox.astype(int)
        face_list.append({
            'facial_area': [bbox[0], bbox[1], bbox[2], bbox[3]],
            'landmarks': {
                'left_eye': face.kps[0].tolist(),
                'right_eye': face.kps[1].tolist(),
                'nose': face.kps[2].tolist(),
                'mouth_left': face.kps[3].tolist(),
                'mouth_right': face.kps[4].tolist()
            },
            'score': float(face.det_score)
        })
    
    return face_list


def compute_face_embedding(face_image: Any) -> Any:
    """
    Compute a numerical embedding vector for the provided face image.

    The embedding should capture discriminative facial features for comparison.
    """
    if isinstance(face_image, bytes):
        face_image = _bytes_to_image(face_image)
    
    app = _get_face_app()
    
    # 정렬된 112x112 이미지인 경우 직접 recognition 모델 사용
    if face_image.shape == (112, 112, 3):
        # Recognition 모델 ONNX 세션에 직접 접근
        rec_model = None
        for model in app.models.values():
            if hasattr(model, 'taskname') and model.taskname == 'recognition':
                rec_model = model
                break
        
        if rec_model is None:
            raise ValueError("Recognition model not found")
        
        # 전처리: BGR -> RGB, 정규화
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_rgb = np.transpose(face_rgb, (2, 0, 1))  # HWC -> CHW
        face_input = np.expand_dims(face_rgb, axis=0).astype(np.float32)
        face_input = (face_input - 127.5) / 127.5
        
        # ONNX 모델에 직접 inference
        input_name = rec_model.session.get_inputs()[0].name
        outputs = rec_model.session.run(None, {input_name: face_input})
        embedding = outputs[0].flatten()
        
        return embedding
    else:
        # 원본 이미지인 경우
        faces = app.get(face_image)
        
        if len(faces) == 0:
            raise ValueError("No face detected for embedding")
        
        embedding = faces[0].embedding
        return embedding

def detect_face_keypoints(face_image: Any) -> Any:
    """
    Identify facial keypoints (landmarks) for alignment or analysis.

    The return type can be tailored to the chosen keypoint detection library.
    """
    faces = detect_faces(face_image)
    
    if not faces:
        return None
    
    best_face = max(faces, key=lambda f: f['score'])
    landmarks = best_face['landmarks']
    
    keypoints = np.array([
        landmarks['left_eye'],
        landmarks['right_eye'],
        landmarks['nose'],
        landmarks['mouth_left'],
        landmarks['mouth_right']
    ], dtype=np.float32)
    
    return keypoints


def warp_face(image: Any, homography_matrix: Any) -> Any:
    """
    Warp the provided face image using the supplied homography matrix.

    Typically used to align faces prior to embedding extraction.
    """
    if isinstance(image, bytes):
        image = _bytes_to_image(image)
    
    result = cv2.estimateAffinePartial2D(
        homography_matrix, 
        ARCFACE_DST,
        method=cv2.LMEDS
    )
    
    if result is None or result[0] is None:
        raise ValueError("Failed to estimate transformation matrix for face alignment")
    
    transformation_matrix = result[0]
    
    aligned_face = cv2.warpAffine(
        image, 
        transformation_matrix, 
        (112, 112),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    return aligned_face


def antispoof_check(face_image: Any) -> float:
    """
    Perform an anti-spoofing check and return a confidence score.

    A higher score should indicate a higher likelihood that the face is real.
    """
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = min(laplacian_var / 500.0, 1.0)
    
    color_std = np.std(face_image)
    color_score = min(color_std / 50.0, 1.0)
    
    confidence = 0.6 * sharpness_score + 0.4 * color_score
    
    return float(confidence)


def calculate_face_similarity(image_a: Any, image_b: Any) -> float:
    """
    End-to-end pipeline that returns a similarity score between two faces.

    This function should:
      1. Detect faces in both images.
      2. Align faces using keypoints and homography warping.
      3. (Run anti-spoofing checks to validate face authenticity. - If you want)
      4. Generate embeddings and compute a similarity score.

    The images provided by the API arrive as raw byte strings; convert or decode
    them as needed for downstream processing.
    """
    # 1. 원본 이미지 변환
    if isinstance(image_a, bytes):
        img_a = _bytes_to_image(image_a)
    else:
        img_a = image_a
        
    if isinstance(image_b, bytes):
        img_b = _bytes_to_image(image_b)
    else:
        img_b = image_b
    
    # 2. 얼굴 keypoint 검출
    keypoints_a = detect_face_keypoints(img_a)
    keypoints_b = detect_face_keypoints(img_b)
    
    if keypoints_a is None:
        raise ValueError("No face detected in image A")
    if keypoints_b is None:
        raise ValueError("No face detected in image B")
    
    # 3. 얼굴 정렬 (anti-spoofing 체크용)
    aligned_a = warp_face(img_a, keypoints_a)
    aligned_b = warp_face(img_b, keypoints_b)
    
    # 4. Anti-spoofing 체크
    spoof_score_a = antispoof_check(aligned_a)
    spoof_score_b = antispoof_check(aligned_b)
    
    if spoof_score_a < 0.3 or spoof_score_b < 0.3:
        print(f"Warning: Low anti-spoof confidence (A: {spoof_score_a:.2f}, B: {spoof_score_b:.2f})")
    
    # 5. 정렬된 얼굴(112x112)에서 embedding 추출
    embedding_a = compute_face_embedding(aligned_a)
    embedding_b = compute_face_embedding(aligned_b)
    
    # 6. 유사도 계산
    embedding_a_2d = embedding_a.reshape(1, -1)
    embedding_b_2d = embedding_b.reshape(1, -1)
    
    similarity = cosine_similarity(embedding_a_2d, embedding_b_2d)[0][0]
    similarity_normalized = (similarity + 1.0) / 2.0
    
    return float(similarity_normalized)


_face_app = None
