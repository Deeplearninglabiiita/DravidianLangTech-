

# pip install torch torchvision pandas pillow transformers tqdm scikit-learn
# pip install git+https://github.com/openai/CLIP.git

import os
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
from transformers import AutoModel, AutoTokenizer, CLIPProcessor, CLIPModel
from tqdm import tqdm
from sklearn.metrics import classification_report


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_csv = '../Tamil/train.csv'
dev_csv = '../Tamil/dev.csv'
test_csv = '../Tamil/test.csv'
train_img_dir = '../Tamil/train'
dev_img_dir = '../Tamil/dev'
test_img_dir = '../Tamil/test'


clip_model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(clip_model_name)
clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)


text_model_name = "ai4bharat/indic-bert"  # IndicBERT supports Tamil
text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
text_encoder = AutoModel.from_pretrained(text_model_name).to(device)


class MultimodalClassifier(torch.nn.Module):
    def __init__(self, text_encoder, clip_model):
        super().__init__()
        self.text_encoder = text_encoder
        self.clip = clip_model

        fused_dim = self.clip.config.projection_dim + self.text_encoder.config.hidden_size
        self.classifier = torch.nn.Linear(fused_dim, 1)

    def forward(self, input_ids, attention_mask, pixel_values):

        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]  # CLS token representation


        image_features = self.clip.get_image_features(pixel_values=pixel_values)


        fused = torch.cat((text_features, image_features), dim=1)
        logits = self.classifier(fused)
        return logits

model = MultimodalClassifier(text_encoder, clip_model).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)


train_dataset = MemeDataset(train_csv, train_img_dir)
dev_dataset = MemeDataset(dev_csv, dev_img_dir)
test_dataset = MemeDataset(test_csv, test_img_dir)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True,
                          collate_fn=lambda b: custom_collate_fn(b, processor, device, include_labels=True))
dev_loader = DataLoader(dev_dataset, batch_size=2, shuffle=False,
                        collate_fn=lambda b: custom_collate_fn(b, processor, device, include_labels=True))
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False,
                         collate_fn=lambda b: custom_collate_fn(b, processor, device, include_labels=False))


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


EPOCHS = 12
for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_preds, val_labels = evaluate(model, dev_loader, criterion)
    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(classification_report(val_labels, val_preds, digits=4))


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


test_data = pd.read_csv(test_csv)
test_data['predicted_labels'] = test_predictions[:len(test_data)]
test_data.to_csv("../Tamil/predictions_indic_Tamil.csv", index=False)
print("Prediction results saved.")

import pandas as pd
df1=pd.read_csv("../Tamil/predictions_indic_Tamil.csv")
df1.head()

label_counts = df1['predicted_labels'].value_counts()

print(label_counts)



import os
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
from transformers import AutoModel, AutoTokenizer, CLIPProcessor, CLIPModel
from tqdm import tqdm
from sklearn.metrics import classification_report


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_csv = '../Malayalam/train.csv'
dev_csv = '../Malayalam/dev.csv'
test_csv = '../Malayalam/test.csv'
train_img_dir = '../Malayalam/train'
dev_img_dir = '../Malayalam/dev'
test_img_dir = '../Malayalam/test'


clip_model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(clip_model_name)
clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)

text_model_name = "ai4bharat/indic-bert"  # IndicBERT supports Tamil
text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
text_encoder = AutoModel.from_pretrained(text_model_name).to(device)


class MultimodalClassifier(torch.nn.Module):
    def __init__(self, text_encoder, clip_model):
        super().__init__()
        self.text_encoder = text_encoder
        self.clip = clip_model

        fused_dim = self.clip.config.projection_dim + self.text_encoder.config.hidden_size
        self.classifier = torch.nn.Linear(fused_dim, 1)

    def forward(self, input_ids, attention_mask, pixel_values):

        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]  # CLS token representation


        image_features = self.clip.get_image_features(pixel_values=pixel_values)


        fused = torch.cat((text_features, image_features), dim=1)
        logits = self.classifier(fused)
        return logits

model = MultimodalClassifier(text_encoder, clip_model).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)


train_dataset = MemeDataset(train_csv, train_img_dir)
dev_dataset = MemeDataset(dev_csv, dev_img_dir)
test_dataset = MemeDataset(test_csv, test_img_dir)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True,
                          collate_fn=lambda b: custom_collate_fn(b, processor, device, include_labels=True))
dev_loader = DataLoader(dev_dataset, batch_size=2, shuffle=False,
                        collate_fn=lambda b: custom_collate_fn(b, processor, device, include_labels=True))
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False,
                         collate_fn=lambda b: custom_collate_fn(b, processor, device, include_labels=False))


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


EPOCHS = 12
for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_preds, val_labels = evaluate(model, dev_loader, criterion)
    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(classification_report(val_labels, val_preds, digits=4))


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


test_data = pd.read_csv(test_csv)
test_data['predicted_labels'] = test_predictions[:len(test_data)]
test_data.to_csv("../Malayalam/predictions_indic_Malayalam.csv", index=False)
print("Prediction results saved.")

#................



