import os
import yaml
import shutil
from ultralytics import YOLO

def main():
    print("=== Starting Custom YOLOv8-seg Model Training Pipeline ===")
    
    # Generate dataset_yolo.yaml config dynamically
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")).replace("\\", "/")
    dataset_dir = f"{ROOT_DIR}/dataset_yolo"
    
    yaml_data = {
        "path": dataset_dir,
        "train": "images/train",
        "val": "images/val",
        "names": {0: "udder"}
    }
    
    yaml_path = os.path.join(ROOT_DIR, "dataset_yolo.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
        
    model = YOLO("yolov8n-seg.pt")
    
    # Train YOLOv8-seg on RTX 3050 GPU
    model.train(
        data=yaml_path,
        epochs=30,
        imgsz=640,
        batch=16,
        device=0,
        workers=0,
        project=os.path.join(ROOT_DIR, "runs", "segment"),
        name="train_udder"
    )
    
    # Save best weights to Model directory
    src = os.path.join(ROOT_DIR, "runs", "segment", "train_udder", "weights", "best.pt")
    dst = os.path.join(ROOT_DIR, "Model", "best_udder_yolo.pt")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    print(f"Model saved to: {dst}")

if __name__ == "__main__":
    main()
