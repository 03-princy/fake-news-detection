# scripts/preprocess.py
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re


nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def load_and_preprocess_data():
    # Load datasets
    fake_news = pd.read_csv('data/Fake.csv')
    true_news = pd.read_csv('data/True.csv')

    # Add labels
    fake_news['label'] = 0  # Fake news
    true_news['label'] = 1  # True news

    # Combine datasets
    data = pd.concat([fake_news, true_news], axis=0)

    # Preprocess text
    data['text'] = data['title'] + ' ' + data['text']
    data['text'] = data['text'].apply(preprocess_text)

    
    data = data.sample(frac=1).reset_index(drop=True)

    return data

if __name__ == '__main__':
    data = load_and_preprocess_data()
    data.to_csv('data/processed_data.csv', index=False)