from ultralytics import YOLO
import torch

def main():
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("No GPU detected. Training will run on CPU.")

    model = YOLO("yolov8m.yaml")  # Initialize model with custom YAML
    model = model.load("yolov8m.pt")  # Load pretrained weights if available

    results = model.train(data="data.yaml", epochs=100, imgsz=512, batch=2, cache=False)

if __name__ == "__main__":
    main()