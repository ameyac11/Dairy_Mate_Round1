---
title: Dairy Mate Mastitis Detection
emoji: 🐄
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "6.19.0"
app_file: app.py
pinned: false
license: apache-2.0
short_description: AI-powered cattle mastitis detection using YOLOv8 + CNN
---

# 🐄 Dairy Mate: Mastitis Diagnostics Dashboard

An end-to-end AI pipeline for automated cattle udder health monitoring.

**How it works:**
1. Upload a cow udder photo
2. YOLOv8-seg isolates the udder region
3. Inflammation hotspots are detected via HSV color thresholding
4. ResNet-50 or MobileNet-V3 classifies the udder as **Healthy** or **Mastitis**

**Models included:**
- `Model/best_udder_yolo.pt` — YOLOv8-seg udder segmentation
- `Model/resnet50_mastitis.safetensors` — ResNet-50 classifier
- `Model/mobilenetv3_mastitis.safetensors` — MobileNet-V3 classifier

Licensed under Apache 2.0.
