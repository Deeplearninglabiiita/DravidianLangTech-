# -*- coding: utf-8 -*-
"""run2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1lsL2P5fLxLwECMwFGilCWh537Ri6LUQe
"""



from google.colab import drive
drive.mount('/content/drive')



pip install torch torchvision pandas pillow transformers tqdm scikit-learn

from google.colab import drive
drive.mount('/content/drive')

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from torchvision import models
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Dataset for labeled data with images and transcriptions
class LabeledMemeDataset(Dataset):
    def __init__(self, csv_file, img_dir, tokenizer, max_length=128, transform=None, sep='\t'):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_id = row['image_id']
        label = row['labels']
        text = row['transcriptions']
        image_path = os.path.join(self.img_dir, f"{image_id}.jpg")

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'image': image,
            'label': torch.tensor(label, dtype=torch.float)
        }

# Dataset for unlabeled test data with images and transcriptions
class UnlabeledMemeDataset(Dataset):
    def __init__(self, csv_file, img_dir, tokenizer, max_length=128, transform=None, sep='\t'):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_id = row['image_id']
        text = row['transcriptions']
        image_path = os.path.join(self.img_dir, f"{image_id}.jpg")

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'image': image
        }

# Image transformation for ResNet
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to data
train_csv = '/content/drive/MyDrive/NAACL_2025/Tamil/train.csv'
dev_csv = '/content/drive/MyDrive/NAACL_2025/Tamil/dev.csv'
test_csv = '/content/drive/MyDrive/NAACL_2025/Tamil/test.csv'
train_img_dir = '/content/drive/MyDrive/NAACL_2025/Tamil/train'
dev_img_dir = '/content/drive/MyDrive/NAACL_2025/Tamil/dev'
test_img_dir = '/content/drive/MyDrive/NAACL_2025/Tamil/test'

# Load MuRIL tokenizer and model for text
text_model_name = "google/muril-large-cased"
tokenizer = AutoTokenizer.from_pretrained(text_model_name)
text_encoder = AutoModel.from_pretrained(text_model_name).to(device)

# Load pretrained ResNet-50 for image encoding
resnet = models.resnet50(pretrained=True)
num_features = resnet.fc.in_features
resnet.fc = nn.Identity()  # Use ResNet as a feature extractor
resnet = resnet.to(device)

# Define multimodal classifier using MuRIL and ResNet
class MultimodalClassifier(nn.Module):
    def __init__(self, text_encoder, image_encoder, dropout_rate=0.3):
        super(MultimodalClassifier, self).__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder

        text_dim = self.text_encoder.config.hidden_size
        image_dim = num_features
        fusion_dim = 256  # Projection dimension for each modality

        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.image_proj = nn.Linear(image_dim, fusion_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(fusion_dim * 2, 1)

    def forward(self, input_ids, attention_mask, images):
        # Text encoding
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feats = text_outputs.last_hidden_state[:, 0, :]  # CLS token
        text_feats = self.text_proj(text_feats)

        # Image encoding
        image_feats = self.image_encoder(images)
        image_feats = self.image_proj(image_feats)

        # Concatenate features and classify
        fused = torch.cat((text_feats, image_feats), dim=1)
        fused = self.dropout(fused)
        logits = self.classifier(fused)
        return logits

model = MultimodalClassifier(text_encoder, resnet, dropout_rate=0.3).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)

# Create datasets and dataloaders
train_dataset = LabeledMemeDataset(train_csv, train_img_dir, tokenizer, max_length=128, transform=image_transform)
dev_dataset = LabeledMemeDataset(dev_csv, dev_img_dir, tokenizer, max_length=128, transform=image_transform)
test_dataset = UnlabeledMemeDataset(test_csv, test_img_dir, tokenizer, max_length=128, transform=image_transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
dev_loader = DataLoader(dev_dataset, batch_size=8, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

# Training loop with tqdm
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        images = batch['image'].to(device)
        labels = batch['label'].unsqueeze(1).to(device)

        outputs = model(input_ids, attention_mask, images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].unsqueeze(1).to(device)

            outputs = model(input_ids, attention_mask, images)
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
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_preds, val_labels = evaluate_model(model, dev_loader, criterion, device)
    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(classification_report(val_labels, val_preds, digits=4))

# Prediction on test set
model.eval()
test_predictions = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        images = batch['image'].to(device)

        outputs = model(input_ids, attention_mask, images)
        preds = (torch.sigmoid(outputs) > 0.5).long().cpu().numpy()
        test_predictions.extend(preds)

# Save predictions to CSV
test_data = pd.read_csv(test_csv)
test_data['predicted_labels'] = test_predictions[:len(test_data)]
test_data.to_csv("/content/drive/MyDrive/NAACL_2025/Tamil/predictions_goog_mur_res.csv", index=False)
print("Prediction results saved.")

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from torchvision import models
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Dataset for labeled data with images and transcriptions
class LabeledMemeDataset(Dataset):
    def __init__(self, csv_file, img_dir, tokenizer, max_length=128, transform=None, sep='\t'):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_id = row['image_id']
        label = row['labels']
        text = row['transcriptions']
        image_path = os.path.join(self.img_dir, f"{image_id}.jpg")

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'image': image,
            'label': torch.tensor(label, dtype=torch.float)
        }

# Dataset for unlabeled test data with images and transcriptions
class UnlabeledMemeDataset(Dataset):
    def __init__(self, csv_file, img_dir, tokenizer, max_length=128, transform=None, sep='\t'):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_id = row['image_id']
        text = row['transcriptions']
        image_path = os.path.join(self.img_dir, f"{image_id}.jpg")

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'image': image
        }

# Image transformation for ResNet
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to data
train_csv = '/content/drive/MyDrive/NAACL_2025/Malayalam/train.csv'
dev_csv = '/content/drive/MyDrive/NAACL_2025/Malayalam/dev.csv'
test_csv = '/content/drive/MyDrive/NAACL_2025/Malayalam/test.csv'
train_img_dir = '/content/drive/MyDrive/NAACL_2025/Malayalam/train'
dev_img_dir = '/content/drive/MyDrive/NAACL_2025/Malayalam/dev'
test_img_dir = '/content/drive/MyDrive/NAACL_2025/Malayalam/test'

# Load MuRIL tokenizer and model for text
text_model_name = "google/muril-large-cased"
tokenizer = AutoTokenizer.from_pretrained(text_model_name)
text_encoder = AutoModel.from_pretrained(text_model_name).to(device)

# Load pretrained ResNet-50 for image encoding
resnet = models.resnet50(pretrained=True)
num_features = resnet.fc.in_features
resnet.fc = nn.Identity()  # Use ResNet as a feature extractor
resnet = resnet.to(device)

# Define multimodal classifier using MuRIL and ResNet
class MultimodalClassifier(nn.Module):
    def __init__(self, text_encoder, image_encoder, dropout_rate=0.3):
        super(MultimodalClassifier, self).__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder

        text_dim = self.text_encoder.config.hidden_size
        image_dim = num_features
        fusion_dim = 256  # Projection dimension for each modality

        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.image_proj = nn.Linear(image_dim, fusion_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(fusion_dim * 2, 1)

    def forward(self, input_ids, attention_mask, images):
        # Text encoding
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feats = text_outputs.last_hidden_state[:, 0, :]  # CLS token
        text_feats = self.text_proj(text_feats)

        # Image encoding
        image_feats = self.image_encoder(images)
        image_feats = self.image_proj(image_feats)

        # Concatenate features and classify
        fused = torch.cat((text_feats, image_feats), dim=1)
        fused = self.dropout(fused)
        logits = self.classifier(fused)
        return logits

model = MultimodalClassifier(text_encoder, resnet, dropout_rate=0.3).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)

# Create datasets and dataloaders
train_dataset = LabeledMemeDataset(train_csv, train_img_dir, tokenizer, max_length=128, transform=image_transform)
dev_dataset = LabeledMemeDataset(dev_csv, dev_img_dir, tokenizer, max_length=128, transform=image_transform)
test_dataset = UnlabeledMemeDataset(test_csv, test_img_dir, tokenizer, max_length=128, transform=image_transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
dev_loader = DataLoader(dev_dataset, batch_size=8, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

# Training loop with tqdm
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        images = batch['image'].to(device)
        labels = batch['label'].unsqueeze(1).to(device)

        outputs = model(input_ids, attention_mask, images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].unsqueeze(1).to(device)

            outputs = model(input_ids, attention_mask, images)
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
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_preds, val_labels = evaluate_model(model, dev_loader, criterion, device)
    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(classification_report(val_labels, val_preds, digits=4))

# Prediction on test set
model.eval()
test_predictions = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        images = batch['image'].to(device)

        outputs = model(input_ids, attention_mask, images)
        preds = (torch.sigmoid(outputs) > 0.5).long().cpu().numpy()
        test_predictions.extend(preds)

# Save predictions to CSV
test_data = pd.read_csv(test_csv)
test_data['predicted_labels'] = test_predictions[:len(test_data)]
test_data.to_csv("/content/drive/MyDrive/NAACL_2025/Malayalam/predictions_goog_mur_res.csv", index=False)
print("Prediction results saved.")