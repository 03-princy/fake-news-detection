from flask import Flask, render_template, request, jsonify
import joblib
import requests
from scripts.utils import preprocess_text

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('models/fake_news_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# NewsAPI configuration
NEWS_API_KEY = ''  # Replace with your NewsAPI key
NEWS_API_URL = 'https://newsapi.org/v2/everything'

# Store chat messages
chat_history = []

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', chat_history=chat_history)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input text from the form
    input_text = request.json['news_text']
    # Preprocess the input text
    text = preprocess_text(input_text)
    # Convert text to TF-IDF features
    text_tfidf = vectorizer.transform([text])
    # Make prediction
    prediction_result = model.predict(text_tfidf)
    prediction = "Real News" if prediction_result[0] == 1 else "Fake News"

    # Fetch real news if the prediction is "Fake News"
    real_news = None
    if prediction == "Fake News":
        real_news = fetch_real_news(input_text)

    # Add the message and prediction to the chat history
    chat_history.append({"user": input_text, "prediction": prediction, "real_news": real_news})

    # Return the prediction and real news as JSON
    return jsonify({"prediction": prediction, "real_news": real_news})

def fetch_real_news(query):
    try:
        # Fetch real news from NewsAPI
        params = {
            'q': query,  # Use the user's input as the search query
            'apiKey': NEWS_API_KEY,
            'pageSize': 1,  # Fetch only 1 article
            'sortBy': 'relevancy',  # Sort by relevancy
        }
        print("Fetching real news for query:", query)  # Debug print
        response = requests.get(NEWS_API_URL, params=params)
        print("API Response Status Code:", response.status_code)  # Debug print
        if response.status_code == 200:
            articles = response.json().get('articles', [])
            print("Articles found:", len(articles))  # Debug print
            if articles:
                print("Fetched real news:", articles[0])  # Debug print
                return {
                    "title": articles[0]['title'],
                    "description": articles[0]['description'],
                    "url": articles[0]['url'],
                }
        print("No real news found for query:", query)  # Debug print
    except Exception as e:
        print("Error fetching real news:", e)  # Debug print
    return None

if __name__ == '__main__':
    app.run(debug=True)
