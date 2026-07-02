import os
import json
import cv2
from segmentation import analyze_image

def annotate_dataset():
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    base_dir = os.path.join(ROOT_DIR, "Dataset")
    healthy_dir = os.path.join(base_dir, "healthy_images")
    mastitis_dir = os.path.join(base_dir, "Mastitis_images")
    
    annotations = []
    categories = [
        {"dir": healthy_dir, "label": "Healthy", "class_id": 0},
        {"dir": mastitis_dir, "label": "Mastitis", "class_id": 1}
    ]
    
    stats = {
        "Healthy": {"count": 0, "total_inflammation": 0.0},
        "Mastitis": {"count": 0, "total_inflammation": 0.0}
    }
    
    for category in categories:
        folder = category["dir"]
        label = category["label"]
        class_id = category["class_id"]
        
        if not os.path.exists(folder):
            continue
            
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.jfif'))]
        for filename in files:
            filepath = os.path.join(folder, filename)
            
            try:
                img = cv2.imread(filepath)
                if img is None:
                    continue
                h, w = img.shape[:2]
                
                _, udder_mask, bbox, _, inflam_index = analyze_image(img)
                
                # Standardize path relative to project root
                relative_path = os.path.relpath(filepath, ROOT_DIR).replace("\\", "/")
                
                annotations.append({
                    "filepath": relative_path,
                    "filename": filename,
                    "label": label,
                    "class_id": class_id,
                    "width": w,
                    "height": h,
                    "bbox": bbox,
                    "inflammation_index": round(inflam_index, 2)
                })
                
                stats[label]["count"] += 1
                stats[label]["total_inflammation"] += inflam_index
                
            except Exception as e:
                print(f"Error {filename}: {e}")
                
    output_path = os.path.join(base_dir, "annotations.json")
    with open(output_path, "w") as f:
        json.dump(annotations, f, indent=4)
        
    print(f"Saved {len(annotations)} annotations to {output_path}")

if __name__ == "__main__":
    annotate_dataset()
