"""
InsightFace 모델을 추출하여 Triton model_repository에 복사하는 스크립트

InsightFace는 이미 ONNX 모델을 내부적으로 사용하므로,
해당 모델들을 찾아서 Triton 저장소로 복사하면 됩니다.
"""
import shutil
from pathlib import Path
from insightface.app import FaceAnalysis

def extract_insightface_models(output_dir: Path):
    """
    InsightFace 모델들을 추출하여 지정된 디렉토리에 복사
    """
    print("[extract] Initializing InsightFace...")
    app = FaceAnalysis(providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    # 모델 정보 출력
    print(f"\n[extract] Found {len(app.models)} models:")
    for name, model in app.models.items():
        print(f"  - {name}")
        if hasattr(model, 'taskname'):
            print(f"    Task: {model.taskname}")
        if hasattr(model, 'model_file'):
            print(f"    File: {model.model_file}")
    
    # Detection 모델 추출
    detector_model = None
    detector_name = None
    for name, model in app.models.items():
        if hasattr(model, 'taskname') and model.taskname == 'detection':
            detector_model = model
            detector_name = name
            break
    
    if detector_model and hasattr(detector_model, 'model_file'):
        src_path = Path(detector_model.model_file)
        dst_dir = output_dir / "face_detector" / "1"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_dir / "model.onnx"
        
        print(f"\n[extract] Copying detection model:")
        print(f"  From: {src_path}")
        print(f"  To:   {dst_path}")
        shutil.copy2(src_path, dst_path)
        print(f"  ✓ Detection model copied successfully!")
    else:
        print("[extract] Warning: Detection model not found!")
    
    # Recognition 모델 추출
    recognition_model = None
    recognition_name = None
    for name, model in app.models.items():
        if hasattr(model, 'taskname') and model.taskname == 'recognition':
            recognition_model = model
            recognition_name = name
            break
    
    if recognition_model and hasattr(recognition_model, 'model_file'):
        src_path = Path(recognition_model.model_file)
        dst_dir = output_dir / "fr_model" / "1"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_dir / "model.onnx"
        
        print(f"\n[extract] Copying recognition model:")
        print(f"  From: {src_path}")
        print(f"  To:   {dst_path}")
        shutil.copy2(src_path, dst_path)
        print(f"  ✓ Recognition model copied successfully!")
    else:
        print("[extract] Warning: Recognition model not found!")
    
    print(f"\n[extract] ✓ Model extraction complete!")
    print(f"[extract] Models saved to: {output_dir}")
    
    # 모델 정보 저장
    info_file = output_dir / "model_info.txt"
    with open(info_file, 'w') as f:
        f.write("InsightFace Models Information\n")
        f.write("=" * 50 + "\n\n")
        
        if detector_model:
            f.write(f"Detection Model: {detector_name}\n")
            f.write(f"  Task: {detector_model.taskname}\n")
            if hasattr(detector_model, 'input_size'):
                f.write(f"  Input size: {detector_model.input_size}\n")
            if hasattr(detector_model, 'input_mean'):
                f.write(f"  Input mean: {detector_model.input_mean}\n")
            if hasattr(detector_model, 'input_std'):
                f.write(f"  Input std: {detector_model.input_std}\n")
            f.write("\n")
        
        if recognition_model:
            f.write(f"Recognition Model: {recognition_name}\n")
            f.write(f"  Task: {recognition_model.taskname}\n")
            if hasattr(recognition_model, 'input_size'):
                f.write(f"  Input size: {recognition_model.input_size}\n")
            if hasattr(recognition_model, 'input_mean'):
                f.write(f"  Input mean: {recognition_model.input_mean}\n")
            if hasattr(recognition_model, 'input_std'):
                f.write(f"  Input std: {recognition_model.input_std}\n")
    
    print(f"[extract] Model info saved to: {info_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract InsightFace models to Triton repository")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model_repository"),
        help="Output directory for Triton model repository"
    )
    args = parser.parse_args()
    
    extract_insightface_models(args.output_dir)

