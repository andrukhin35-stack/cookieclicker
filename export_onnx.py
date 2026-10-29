"""Export the best YOLOv8 checkpoint; no training or automatic installs."""
import os
from pathlib import Path


def main():
    import onnx
    from ultralytics import YOLO
    root = Path(__file__).resolve().parent
    os.chdir(root)
    model = YOLO(root / "runs/detect/cookie_detector/weights/best.pt")
    result = model.export(format="onnx", imgsz=640, opset=12, dynamic=False,
                          simplify=False, half=False, nms=False, device="cpu")
    onnx.checker.check_model(onnx.load(result))
    print(f"ONNX ready: {result}")


if __name__ == "__main__":
    main()
