# Fake & Real News Detection 📰🤖

This project is a **Machine Learning web application** that detects whether a news article is **Fake** or **Real** using **Natural Language Processing (NLP)**.

The project follows a clear pipeline:
1. Concatenate datasets into a single CSV file
2. Clean and preprocess text data
3. Train a machine learning model
4. Predict results via a web interface

---

## 📂 Project Structure
```
fake_real_news_ML/
├── data/
│ ├── Fake/
│ │ └── Fake.csv
│ ├── True/
│ │ └── True.csv
│ └── news.csv # Generated after concatenation
├── model/
│ ├── concat.py # Merge Fake & True datasets
│ ├── cleanNLP.py # Text cleaning & preprocessing
│ └── train.py # Model training
├── static/
│ ├── img/
│ │ └── codiia.png
│ ├── index.css
│ ├── index.html
│ └── script.js
├── app.py # FastAPI application
├── requirements.txt
└── README.md
```
---

## 🗂 Datasets

The datasets used in this project come from **Kaggle (2017)**:

- Fake news: `data/Fake/Fake.csv`
- Real news: `data/True/True.csv`

**Source:** [Fake and Real News Dataset on Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

**Note:** The two CSV files are concatenated into a single file `data/news.csv` via `model/concat.py`.

---

## 🔄 Machine Learning Workflow

### 1️⃣ Data Concatenation
- Fake and Real news datasets are merged
- Output file: `data/news.csv`
- Script: `model/concat.py`

### 2️⃣ Text Cleaning (NLP)
- Cleaning and preprocessing of news text
- Implemented in: `model/cleanNLP.py`
- Uses regular expressions and NLP techniques

### 3️⃣ Model Training
- TF-IDF vectorization
- Logistic Regression classifier
- Model training and evaluation
- Script: `model/train.py`

---

## 📚 Libraries Used

```python
import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

```
---
🌐 Web Application

Framework: FastAPI

Backend file: app.py

Frontend: HTML, CSS, JavaScript

The user enters a news article and receives a prediction:

Fake

Real
---
⚙️ Installation

git clone https://github.com/AkramLamfaddel/fake-news-detector.git
cd fake_real_news_ML
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
---
▶️ Run the Project

Step 1: Concatenate datasets

```python model/concat.py```

Step 2: Clean text data

```python model/cleanNLP.py```

Step 3: Train the model

```python model/train.py```

Step 4: Run the web app

```uvicorn app:app --reload```

Open your browser:

 http://127.0.0.1:8000 

🚀 Future Improvements

Improve model accuracy

Add deep learning models

Deploy the application online

Enhance UI/UX

👤 Author

Akram Lamfaddel Codi-IA

