# 📰 Fake News Detection System

**Detect • Prevent • Inform**

A machine learning-based system to classify news articles as **fake** or **real**, helping curb the spread of misinformation using NLP techniques and deep learning models.

---

## ✨ Features Overview

| Feature               | Description                                                     | Technology                 |
| --------------------- | --------------------------------------------------------------- | -------------------------- |
| 📄 Data Preprocessing | Cleaning, tokenizing, and vectorizing news text                 | NLTK, Scikit-learn         |
| 🤖 ML & DL Models     | Includes Logistic Regression, PassiveAggressiveClassifier, BERT | Scikit-learn, Transformers |
| 📊 Model Evaluation   | Accuracy, confusion matrix, classification report               | Matplotlib, Seaborn        |
| 🧠 BERT Integration   | Transformer-based deep learning for advanced NLP classification | HuggingFace Transformers   |
| 🧪 Jupyter Notebooks  | Well-structured EDA and model training/testing notebooks        | Jupyter Notebook           |
| 🖥️ Web Interface     | Minimal Flask web app for real-time prediction                  | Flask, HTML, CSS           |

---

## 🗂️ Project Structure

```
fake-news-detection/
│
├── app.py                 # Flask application
├── requirements.txt       # Python dependencies
├── templates/             # HTML templates for web app
├── scripts/               # Model training and utility scripts
├── notebooks/             # EDA and model training Jupyter notebooks
├── models/                # Saved ML and BERT models
├── data/                  # Dataset files (Fake.csv, True.csv)
└── README.md              # Project overview
```

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/03-princy/fake-news-detection.git
cd fake-news-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
python app.py
```

Access the app at: [http://localhost:5000](http://localhost:5000)

---

## 📊 Dataset

The project uses the **Kaggle Fake News Dataset**, containing:

* `True.csv`: Real news articles
* `Fake.csv`: Fake news articles

### Preprocessing Steps:

* Lowercasing
* Stopword Removal
* Lemmatization
* Vectorization: TF-IDF / BERT Embeddings

---

## 🤖 Models Used

| Model                              | Accuracy |
| ---------------------------------- | -------- |
| Logistic Regression                | \~95%    |
| PassiveAggressiveClassifier        | \~94%    |
| BERT (fine-tuned)                  | \~97%    |
| Multinomial Naive Bayes (optional) | \~93%    |

---

## 🙋‍♀️ Author

**Priyanka Singh**
👩‍💻 B.Sc. CS & IT

🔗 [GitHub Profile](https://github.com/03-princy)

💼 [LinkedIn](https://www.linkedin.com/in/priyanka-singh-aa270123a/)

📧 *Connect for collaborations and research!*

---

## 🛡️ License

This project is licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for details.

---

## ⭐ Acknowledgments

* [Scikit-learn](https://scikit-learn.org/)
* [HuggingFace Transformers](https://huggingface.co/transformers/)
* [Kaggle Fake News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
