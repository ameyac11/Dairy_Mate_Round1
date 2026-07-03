import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from safetensors.torch import save_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ANNOTATIONS_PATH = os.path.join(ROOT_DIR, "Dataset", "annotations.json")
MODEL_DIR = os.path.join(ROOT_DIR, "Model")

# Our dataset has only 100 images (25 healthy / 75 mastitis) after the 80/20
# train-val split, so we tune these conservatively to avoid overfitting:
# --------------------------------------------------------------------------
num_epochs = 30
batch_size = 16
learning_rate = 0.0001
seed = 42

random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

class IndexedMastitisDataset(Dataset):
    def __init__(self, annotations, indices, transform=None):
        self.annotations = [annotations[i] for i in indices]
        self.transform = transform
        
    def __len__(self):
        return len(self.annotations)
        
    def __getitem__(self, idx):
        item = self.annotations[idx]
        img_path = os.path.join(ROOT_DIR, item['filepath'])
        img = Image.open(img_path).convert('RGB')
        
        # Crop to udder bounding box
        x, y, w, h = item['bbox']
        if w > 0 and h > 0:
            img = img.crop((x, y, x + w, y + h))
            
        if self.transform:
            img = self.transform(img)
            
        return img, int(item['class_id'])

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

with open(ANNOTATIONS_PATH, 'r') as f:
    annotations = json.load(f)

indices = list(range(len(annotations)))
random.shuffle(indices)
split_idx = int(0.8 * len(annotations))

train_loader = DataLoader(
    IndexedMastitisDataset(annotations, indices[:split_idx], transform=train_transform),
    batch_size=batch_size, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    IndexedMastitisDataset(annotations, indices[split_idx:], transform=val_transform),
    batch_size=batch_size, shuffle=False, num_workers=0
)

def get_model(model_name):
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif model_name == "mobilenetv3":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    return model.to(device)

def train(model_name):
    print(f"\n--- Training {model_name} ---")
    model = get_model(model_name)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        train_acc = 100 * correct_train / total_train
        
        # Validation
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        val_acc = 100 * correct_val / total_val
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss/len(train_loader):.4f} - Train: {train_acc:.1f}% - Val: {val_acc:.1f}%")
        
        if val_acc >= best_acc:
            best_acc = val_acc
            os.makedirs(MODEL_DIR, exist_ok=True)
            save_model(model, os.path.join(MODEL_DIR, f"{model_name}_mastitis.safetensors"))
            print(f"  [SAVED] Accuracy: {best_acc:.1f}%")

if __name__ == "__main__":
    train("mobilenetv3")
    train("resnet50")
