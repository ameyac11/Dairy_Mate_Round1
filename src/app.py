import warnings
warnings.filterwarnings("ignore")
import sys
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from safetensors.torch import load_model
import cv2
from PIL import Image
import gradio as gr

sys.path.insert(0, os.path.dirname(__file__))
from segmentation import analyze_image, generate_visualization

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2
classes = ["Healthy", "Mastitis"]

# Load ResNet-50
resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
resnet50.fc = nn.Linear(resnet50.fc.in_features, num_classes)
try:
    load_model(resnet50, os.path.join(ROOT_DIR, "Model", "resnet50_mastitis.safetensors"))
except Exception:
    pass
resnet50 = resnet50.to(device).eval()

# Load MobileNet-V3
mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
mobilenet.classifier[3] = nn.Linear(mobilenet.classifier[3].in_features, num_classes)
try:
    load_model(mobilenet, os.path.join(ROOT_DIR, "Model", "mobilenetv3_mastitis.safetensors"))
except Exception:
    pass
mobilenet = mobilenet.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image, model_choice):
    if image is None:
        return "No image provided", {}, None, None, None, 0.0
    try:
        image_bgr, udder_mask, bbox, inflam_mask, inflam_index = analyze_image(image)
        x, y, w, h = bbox
        img_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        img_cropped = img_pil.crop((x, y, x + w, y + h)) if w > 0 and h > 0 else img_pil
        img_tensor = transform(img_cropped).unsqueeze(0).to(device)

        selected_model = resnet50 if model_choice == "ResNet-50" else mobilenet
        with torch.no_grad():
            output = selected_model(img_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            predicted_idx = torch.argmax(probabilities).item()

        label = classes[predicted_idx]
        confidence = {classes[i]: float(probabilities[i]) for i in range(num_classes)}
        img_bbox, img_seg, img_overlay, _ = generate_visualization(image)
        return label, confidence, img_bbox, img_seg, img_overlay, round(inflam_index, 2)
    except Exception as e:
        return f"Error: {e}", {}, None, None, None, 0.0

example_images = [
    [os.path.join(ROOT_DIR, "examples", "healthy_example_1.jpg"), "MobileNet-V3"],
    [os.path.join(ROOT_DIR, "examples", "healthy_example_2.jpg"), "ResNet-50"],
    [os.path.join(ROOT_DIR, "examples", "healthy_example_3.jpg"), "MobileNet-V3"],
    [os.path.join(ROOT_DIR, "examples", "mastitis_example_1.jpg"), "ResNet-50"],
    [os.path.join(ROOT_DIR, "examples", "mastitis_example_2.jpg"), "MobileNet-V3"],
    [os.path.join(ROOT_DIR, "examples", "mastitis_example_3.jpg"), "ResNet-50"]
]

with gr.Blocks(title="Dairy Mate - Mastitis Detection") as demo:
    gr.Markdown("# 🐄 Dairy Mate: Mastitis Diagnostics Dashboard")
    gr.Markdown("Upload a cow udder photo to detect mastitis using deep learning segmentation and classification.")

    with gr.Row():
        with gr.Column(scale=5):
            image_input = gr.Image(label="Upload Udder Photo", type="numpy")
            model_selector = gr.Dropdown(choices=["ResNet-50", "MobileNet-V3"], value="MobileNet-V3", label="Model")
            predict_btn = gr.Button("🔍 Run Diagnostics", variant="primary")

        with gr.Column(scale=6):
            with gr.Tabs():
                with gr.TabItem("📋 Health Status"):
                    label_output = gr.Label(label="Diagnosis")
                    confidence_output = gr.Label(label="Confidence", num_top_classes=num_classes)
                    inflammation_slider = gr.Slider(minimum=0.0, maximum=100.0, label="Inflammation Index (%)", interactive=False)
                    gr.Markdown("*Inflammation Index > 1.0% indicates increased mastitis risk.*")

                with gr.TabItem("🖼️ Computer Vision Pipeline"):
                    with gr.Row():
                        bbox_output = gr.Image(label="1. Udder Localization", interactive=False)
                        segmented_output = gr.Image(label="2. Segmented Foreground", interactive=False)
                    overlay_output = gr.Image(label="3. Inflammation Hotspot Mapping", interactive=False)

    gr.Examples(
        examples=example_images,
        inputs=[image_input, model_selector],
        outputs=[label_output, confidence_output, bbox_output, segmented_output, overlay_output, inflammation_slider],
        fn=predict,
        cache_examples=False,
        label="Example Cases"
    )

    predict_btn.click(
        fn=predict,
        inputs=[image_input, model_selector],
        outputs=[label_output, confidence_output, bbox_output, segmented_output, overlay_output, inflammation_slider]
    )

if __name__ == "__main__":
    demo.launch()
