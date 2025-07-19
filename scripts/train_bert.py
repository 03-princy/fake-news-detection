import pandas as pd
import os
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Debugging: Print current working directory and data path
print("Current Working Directory:", os.getcwd())
print("Fake.csv Path:", os.path.abspath('../data/Fake.csv'))
print("True.csv Path:", os.path.abspath('../data/True.csv'))

# Load dataset
try:
    fake_news = pd.read_csv('../data/Fake.csv', encoding='utf-8')
    true_news = pd.read_csv('../data/True.csv', encoding='utf-8')
    
    # Debug: Print the first few rows of each dataset
    print("Fake News Sample:")
    print(fake_news.head())
    
    print("True News Sample:")
    print(true_news.head())
    
    # Use 'title' if 'text' column is missing
    if 'text' not in fake_news.columns:
        fake_news['text'] = fake_news['title']
    if 'text' not in true_news.columns:
        true_news['text'] = true_news['title']
    
except FileNotFoundError as e:
    print("Error: File not found. Please check the file path.")
    raise e
except Exception as e:
    print("Error loading CSV files:", e)
    raise e

# Add labels
fake_news['label'] = 0  # Fake news
true_news['label'] = 1  # Real news

# Combine datasets
data = pd.concat([fake_news, true_news], axis=0)

# Split data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Use a smaller subset for testing
train_data = train_data.sample(frac=0.1, random_state=42)  # Use 10% of the data
test_data = test_data.sample(frac=0.1, random_state=42)    # Use 10% of the data

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize data
def tokenize_data(texts, labels, max_length=128):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    return encodings, torch.tensor(labels)

train_encodings, train_labels = tokenize_data(train_data['text'].tolist(), train_data['label'].tolist())
test_encodings, test_labels = tokenize_data(test_data['text'].tolist(), test_data['label'].tolist())

# Create PyTorch dataset
class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(train_encodings, train_labels)
test_dataset = NewsDataset(test_encodings, test_labels)

# Load BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Training setup
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using device:", device)
model.to(device)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)  # Reduced batch size
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    for i, batch in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f"Epoch {epoch + 1}, Batch {i}, Loss: {loss.item()}")


model.save_pretrained('../models/bert-fake-news')  
tokenizer.save_pretrained('../models/bert-fake-news')  