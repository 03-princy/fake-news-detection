# scripts/predict.py
import tensorflow as tf
import joblib
from utils import preprocess_text

# Load the LSTM model and tokenizer
model = tf.keras.models.load_model('models/fake_news_lstm.h5')
tokenizer = joblib.load('models/tokenizer.pkl')

def predict_news(text):
    
    text = preprocess_text(text)
    
    text_seq = tokenizer.texts_to_sequences([text])
    
    text_pad = tf.keras.preprocessing.sequence.pad_sequences(text_seq, maxlen=200)
    
    prediction = model.predict(text_pad)
    return "Real News" if prediction[0] > 0.5 else "Fake News"

if __name__ == '__main__':
    input_text = input("Enter the news text: ")
    result = predict_news(input_text)
    print("Prediction:", result)