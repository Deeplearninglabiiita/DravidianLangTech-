# -*- coding: utf-8 -*-
"""indic_vit.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1DxaLRL-6gsXWdMD2NOVPesZJVy208xUn
"""

# prompt: mount

from google.colab import drive
drive.mount('/content/drive')

!pip install torch torchvision pandas pillow transformers tqdm scikit-learn
!pip install git+https://github.com/openai/CLIP.git

import os
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
from transformers import AutoModel, AutoTokenizer, CLIPProcessor, CLIPModel
from tqdm import tqdm
from sklearn.metrics import classification_report

# Dataset for labeled data with images and transcriptions
class MemeDataset(Dataset):
    def __init__(self, csv_file, img_dir, sep='\t'):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_id = row['image_id']
        label = row['labels'] if 'labels' in row else None
        text = row['transcriptions']
        image_path = os.path.join(self.img_dir, f"{image_id}.jpg")
        image = Image.open(image_path).convert('RGB')
        sample = {"image": image, "text": text}
        if label is not None:
            sample["label"] = torch.tensor(label, dtype=torch.float)
        return sample

# Custom collate function for batching and processing using CLIPProcessor
def custom_collate_fn(batch, processor, device, include_labels=True):
    texts = [item['text'] for item in batch]
    images = [item['image'] for item in batch]
    labels = None
    if include_labels and 'label' in batch[0]:
        labels = torch.stack([item['label'] for item in batch])

    inputs = processor(text=texts, images=images, return_tensors="pt",
                       padding=True, truncation=True, max_length=77)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if labels is not None:
        inputs['labels'] = labels.to(device)
    return inputs

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to data (update these with your actual paths)
train_csv = '/content/drive/MyDrive/Tamil/train.csv'
dev_csv = '/content/drive/MyDrive/Tamil/dev.csv'
test_csv = '/content/drive/MyDrive/Tamil/test.csv'
train_img_dir = '/content/drive/MyDrive/Tamil/train'
dev_img_dir = '/content/drive/MyDrive/Tamil/dev'
test_img_dir = '/content/drive/MyDrive/Tamil/test'

# Load CLIP processor and model for images
clip_model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(clip_model_name)
clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)

# Load IndicBERT tokenizer and model for Tamil text
text_model_name = "ai4bharat/indic-bert"  # IndicBERT supports Tamil
text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
text_encoder = AutoModel.from_pretrained(text_model_name).to(device)

# Classification model combining IndicBERT and CLIP
class MultimodalClassifier(torch.nn.Module):
    def __init__(self, text_encoder, clip_model):
        super().__init__()
        self.text_encoder = text_encoder
        self.clip = clip_model
        # Fusion dimension: sum of image and text embedding sizes
        fused_dim = self.clip.config.projection_dim + self.text_encoder.config.hidden_size
        self.classifier = torch.nn.Linear(fused_dim, 1)

    def forward(self, input_ids, attention_mask, pixel_values):
        # Extract text features using IndicBERT
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]  # CLS token representation

        # Extract image features using CLIP
        image_features = self.clip.get_image_features(pixel_values=pixel_values)

        # Concatenate text and image features
        fused = torch.cat((text_features, image_features), dim=1)
        logits = self.classifier(fused)
        return logits

model = MultimodalClassifier(text_encoder, clip_model).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Create datasets and dataloaders
train_dataset = MemeDataset(train_csv, train_img_dir)
dev_dataset = MemeDataset(dev_csv, dev_img_dir)
test_dataset = MemeDataset(test_csv, test_img_dir)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True,
                          collate_fn=lambda b: custom_collate_fn(b, processor, device, include_labels=True))
dev_loader = DataLoader(dev_dataset, batch_size=2, shuffle=False,
                        collate_fn=lambda b: custom_collate_fn(b, processor, device, include_labels=True))
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False,
                         collate_fn=lambda b: custom_collate_fn(b, processor, device, include_labels=False))

# Training loop with tqdm
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        pixel_values = batch['pixel_values']
        labels = batch['labels'].unsqueeze(1)

        outputs = model(input_ids, attention_mask, pixel_values)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            pixel_values = batch['pixel_values']
            labels = batch['labels'].unsqueeze(1)

            outputs = model(input_ids, attention_mask, pixel_values)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader)
    return avg_loss, all_preds, all_labels

# Training and evaluation process
EPOCHS = 12
for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_preds, val_labels = evaluate(model, dev_loader, criterion)
    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(classification_report(val_labels, val_preds, digits=4))

# Prediction on test set
model.eval()
test_predictions = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting", leave=False):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        pixel_values = batch['pixel_values']

        outputs = model(input_ids, attention_mask, pixel_values)
        preds = (torch.sigmoid(outputs) > 0.5).long().cpu().numpy()
        test_predictions.extend(preds)

# Save predictions to CSV
test_data = pd.read_csv(test_csv)
test_data['predicted_labels'] = test_predictions[:len(test_data)]
test_data.to_csv("/content/drive/MyDrive/Tamil/predictions_indic.csv", index=False)
print("Prediction results saved.")

import pandas as pd
df1=pd.read_csv("/content/drive/MyDrive/Tamil/predictions_indic.csv")
df1.head()
# Assuming df1 is already defined and loaded with data
label_counts = df1['predicted_labels'].value_counts()

print(label_counts)

from google.colab import drive
drive.mount('/content/drive')

import os
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
from transformers import AutoModel, AutoTokenizer, CLIPProcessor, CLIPModel
from tqdm import tqdm
from sklearn.metrics import classification_report

# Dataset for labeled data with images and transcriptions
class MemeDataset(Dataset):
    def __init__(self, csv_file, img_dir, sep='\t'):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_id = row['image_id']
        label = row['labels'] if 'labels' in row else None
        text = row['transcriptions']
        image_path = os.path.join(self.img_dir, f"{image_id}.jpg")
        image = Image.open(image_path).convert('RGB')
        sample = {"image": image, "text": text}
        if label is not None:
            sample["label"] = torch.tensor(label, dtype=torch.float)
        return sample

# Custom collate function for batching and processing using CLIPProcessor
def custom_collate_fn(batch, processor, device, include_labels=True):
    texts = [item['text'] for item in batch]
    images = [item['image'] for item in batch]
    labels = None
    if include_labels and 'label' in batch[0]:
        labels = torch.stack([item['label'] for item in batch])

    inputs = processor(text=texts, images=images, return_tensors="pt",
                       padding=True, truncation=True, max_length=77)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if labels is not None:
        inputs['labels'] = labels.to(device)
    return inputs

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to data (update these with your actual paths)
train_csv = '/content/drive/MyDrive/Malayalam/train.csv'
dev_csv = '/content/drive/MyDrive/Malayalam/dev.csv'
test_csv = '/content/drive/MyDrive/Malayalam/test.csv'
train_img_dir = '/content/drive/MyDrive/Malayalam/train'
dev_img_dir = '/content/drive/MyDrive/Malayalam/dev'
test_img_dir = '/content/drive/MyDrive/Malayalam/test'

# Load CLIP processor and model for images
clip_model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(clip_model_name)
clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)

# Load IndicBERT tokenizer and model for Tamil text
text_model_name = "ai4bharat/indic-bert"  # IndicBERT supports Tamil
text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
text_encoder = AutoModel.from_pretrained(text_model_name).to(device)

# Classification model combining IndicBERT and CLIP
class MultimodalClassifier(torch.nn.Module):
    def __init__(self, text_encoder, clip_model):
        super().__init__()
        self.text_encoder = text_encoder
        self.clip = clip_model
        # Fusion dimension: sum of image and text embedding sizes
        fused_dim = self.clip.config.projection_dim + self.text_encoder.config.hidden_size
        self.classifier = torch.nn.Linear(fused_dim, 1)

    def forward(self, input_ids, attention_mask, pixel_values):
        # Extract text features using IndicBERT
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]  # CLS token representation

        # Extract image features using CLIP
        image_features = self.clip.get_image_features(pixel_values=pixel_values)

        # Concatenate text and image features
        fused = torch.cat((text_features, image_features), dim=1)
        logits = self.classifier(fused)
        return logits

model = MultimodalClassifier(text_encoder, clip_model).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Create datasets and dataloaders
train_dataset = MemeDataset(train_csv, train_img_dir)
dev_dataset = MemeDataset(dev_csv, dev_img_dir)
test_dataset = MemeDataset(test_csv, test_img_dir)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True,
                          collate_fn=lambda b: custom_collate_fn(b, processor, device, include_labels=True))
dev_loader = DataLoader(dev_dataset, batch_size=2, shuffle=False,
                        collate_fn=lambda b: custom_collate_fn(b, processor, device, include_labels=True))
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False,
                         collate_fn=lambda b: custom_collate_fn(b, processor, device, include_labels=False))

# Training loop with tqdm
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        pixel_values = batch['pixel_values']
        labels = batch['labels'].unsqueeze(1)

        outputs = model(input_ids, attention_mask, pixel_values)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            pixel_values = batch['pixel_values']
            labels = batch['labels'].unsqueeze(1)

            outputs = model(input_ids, attention_mask, pixel_values)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader)
    return avg_loss, all_preds, all_labels

# Training and evaluation process
EPOCHS = 12
for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_preds, val_labels = evaluate(model, dev_loader, criterion)
    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(classification_report(val_labels, val_preds, digits=4))

# Prediction on test set
model.eval()
test_predictions = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting", leave=False):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        pixel_values = batch['pixel_values']

        outputs = model(input_ids, attention_mask, pixel_values)
        preds = (torch.sigmoid(outputs) > 0.5).long().cpu().numpy()
        test_predictions.extend(preds)

# Save predictions to CSV
test_data = pd.read_csv(test_csv)
test_data['predicted_labels'] = test_predictions[:len(test_data)]
test_data.to_csv("/content/drive/MyDrive/Malayalam/predictions_indic.csv", index=False)
print("Prediction results saved.")

#................



