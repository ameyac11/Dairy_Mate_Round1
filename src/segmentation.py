import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Get workspace root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
YOLO_PATH = os.path.join(ROOT_DIR, "Model", "best_udder_yolo.pt")

# Load YOLO model
yolo_model = None
try:
    yolo_model = YOLO(YOLO_PATH)
except Exception:
    pass

def load_image_as_numpy(image_input):
    # Convert image input to BGR numpy array
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_input}")
        return img
    elif isinstance(image_input, Image.Image):
        img_np = np.array(image_input)
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        elif img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        else:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        return img_np
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError("Unsupported image type")

def segment_udder(image_bgr):
    h, w = image_bgr.shape[:2]
    
    # Try YOLO segmentation
    if yolo_model is not None:
        try:
            results = yolo_model(image_bgr, verbose=False)
            for r in results:
                if r.masks is not None and len(r.masks.data) > 0:
                    binary_mask = r.masks.data[0].cpu().numpy().astype(np.uint8) * 255
                    if binary_mask.shape[:2] != (h, w):
                        binary_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    box = r.boxes.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, box)
                    return binary_mask, [x1, y1, x2 - x1, y2 - y1]
        except Exception:
            pass
            
    # Fallback to HSV skin filter + Otsu
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_skin1 = np.array([0, 12, 40])
    upper_skin1 = np.array([28, 255, 255])
    lower_skin2 = np.array([145, 12, 40])
    upper_skin2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
    mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
    skin_mask = cv2.bitwise_or(mask1, mask2)
    
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    combined = cv2.bitwise_and(otsu_thresh, skin_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    udder_mask = np.zeros((h, w), dtype=np.uint8)
    bbox = [0, 0, w, h]
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(udder_mask, [largest_contour], -1, 255, -1)
        x, y, cw, ch = cv2.boundingRect(largest_contour)
        bbox = [x, y, cw, ch]
    else:
        contours_otsu, _ = cv2.findContours(otsu_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_otsu:
            largest_contour = max(contours_otsu, key=cv2.contourArea)
            cv2.drawContours(udder_mask, [largest_contour], -1, 255, -1)
            x, y, cw, ch = cv2.boundingRect(largest_contour)
            bbox = [x, y, cw, ch]
        else:
            udder_mask.fill(255)
            
    return udder_mask, bbox

def detect_inflammation(image_bgr, udder_mask):
    # Detect red inflammation regions on the udder
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 40, 50])
    upper_red1 = np.array([12, 255, 255])
    lower_red2 = np.array([165, 40, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    inflammation_mask = cv2.bitwise_and(red_mask, udder_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(inflammation_mask, cv2.MORPH_OPEN, kernel)

def analyze_image(image_input):
    # Run full segmentation and calculate metrics
    image_bgr = load_image_as_numpy(image_input)
    udder_mask, bbox = segment_udder(image_bgr)
    inflam_mask = detect_inflammation(image_bgr, udder_mask)
    
    udder_area = np.sum(udder_mask > 0)
    inflam_area = np.sum(inflam_mask > 0)
    
    inflam_index = 0.0
    if udder_area > 0:
        inflam_index = (float(inflam_area) / float(udder_area)) * 100.0
        
    return image_bgr, udder_mask, bbox, inflam_mask, inflam_index

def generate_visualization(image_input):
    # Draw green bounding box and red inflammation hotspots
    img_bgr, udder_mask, bbox, inflam_mask, inflam_index = analyze_image(image_input)
    
    img_bbox = img_bgr.copy()
    x, y, w, h = bbox
    cv2.rectangle(img_bbox, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.putText(img_bbox, "Udder Region", (x, max(30, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    img_segmented = cv2.bitwise_and(img_bgr, img_bgr, mask=udder_mask)
    overlay = img_segmented.copy()
    overlay[inflam_mask > 0] = [0, 0, 255]
    img_overlay = cv2.addWeighted(img_segmented, 0.7, overlay, 0.3, 0)
    
    contours, _ = cv2.findContours(inflam_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_overlay, contours, -1, (0, 0, 255), 2)
    
    img_bbox_rgb = cv2.cvtColor(img_bbox, cv2.COLOR_BGR2RGB)
    img_segmented_rgb = cv2.cvtColor(img_segmented, cv2.COLOR_BGR2RGB)
    img_overlay_rgb = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
    
    return (
        Image.fromarray(img_bbox_rgb),
        Image.fromarray(img_segmented_rgb),
        Image.fromarray(img_overlay_rgb),
        inflam_index
    )
