import os
import shutil
import random
import numpy as np
import cv2
import torch
from PIL import Image
from tqdm import tqdm
import argparse
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import SamModel, SamProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up absolute path directories relative to project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HEALTHY_DIR = os.path.join(ROOT_DIR, "Dataset", "healthy_images")
MASTITIS_DIR = os.path.join(ROOT_DIR, "Dataset", "Mastitis_images")
OUTPUT_YOLO_DIR = os.path.join(ROOT_DIR, "dataset_yolo")

os.makedirs(os.path.join(OUTPUT_YOLO_DIR, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_YOLO_DIR, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_YOLO_DIR, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_YOLO_DIR, "labels", "val"), exist_ok=True)

# Load zero-shot models
dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(device)
sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

def auto_segment_image(image_path):
    img_pil = Image.open(image_path).convert("RGB")
    w, h = img_pil.size
    
    # Run Grounding DINO to find the udder bounding box
    inputs = dino_processor(images=img_pil, text="udder .", return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dino_model(**inputs)
        
    results = dino_processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=0.25, text_threshold=0.25, target_sizes=[(h, w)]
    )[0]
    
    boxes = results["boxes"].cpu().numpy()
    if len(boxes) > 0:
        box = boxes[np.argmax(results["scores"].cpu().numpy())]
    else:
        box = np.array([w * 0.15, h * 0.35, w * 0.75, h * 0.80]) # Fallback region
        
    # Run Segment Anything (SAM) on the bounding box
    inputs_sam = sam_processor(img_pil, input_boxes=[[box.tolist()]], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs_sam = sam_model(**inputs_sam)
        
    masks = sam_processor.post_process_masks(
        outputs_sam.pred_masks.cpu(), inputs_sam.original_sizes.cpu(), inputs_sam.reshaped_input_sizes.cpu()
    )[0]
    
    best_mask = masks[0, np.argmax(outputs_sam.iou_scores[0, 0].cpu().numpy())].numpy().astype(np.uint8) * 255
    contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        polygon_points = [[box[0]/w, box[1]/h], [box[2]/w, box[1]/h], [box[2]/w, box[3]/h], [box[0]/w, box[3]/h]]
    else:
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.001 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        polygon_points = [[pt[0]/w, pt[1]/h] for pt in approx.reshape(-1, 2)]
        
    return polygon_points, box

def run_auto_annotation(test_only=False):
    all_images = []
    for folder, label in [(HEALTHY_DIR, "Healthy"), (MASTITIS_DIR, "Mastitis")]:
        if not os.path.exists(folder):
            continue
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.jfif'))]
        for f in files:
            all_images.append({"filename": f, "src_path": os.path.join(folder, f), "label": label})
            
    if test_only:
        if all_images:
            points, box = auto_segment_image(all_images[0]["src_path"])
            print(f"Test Success: Box {box}, {len(points)} polygon points.")
        return
        
    # Split 80/20
    random.seed(42)
    random.shuffle(all_images)
    split_idx = int(0.8 * len(all_images))
    
    splits = [("train", all_images[:split_idx]), ("val", all_images[split_idx:])]
    for split_name, dataset in splits:
        print(f"Processing {split_name} split...")
        for item in tqdm(dataset):
            file_base, ext = os.path.splitext(item["filename"])
            new_base = f"{item['label'].lower()}_{file_base}"
            
            dest_img = os.path.join(OUTPUT_YOLO_DIR, "images", split_name, f"{new_base}{ext}")
            dest_lbl = os.path.join(OUTPUT_YOLO_DIR, "labels", split_name, f"{new_base}.txt")
            
            try:
                points, _ = auto_segment_image(item["src_path"])
                shutil.copy2(item["src_path"], dest_img)
                
                label_data = "0 " + " ".join([f"{pt[0]:.6f} {pt[1]:.6f}" for pt in points])
                with open(dest_lbl, "w") as f:
                    f.write(label_data)
            except Exception as e:
                print(f"Error {item['filename']}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    run_auto_annotation(test_only=args.test)
