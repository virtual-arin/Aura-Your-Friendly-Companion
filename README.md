# Aura - Your Friendly Companion 🤖 | NLP-Based Chatbot for Data Science Guidance

## 🚀 Overview
Aura is a lightweight deep learning chatbot built using TensorFlow and NLP techniques. It classifies user queries into specific intents to provide curated guidance for beginners navigating the Data Science landscape.

## 🧠 Key Features
* **Intent Classification:** Powered by a deep learning backend to categorize user queries.
* **Text Preprocessing:** Implements Tokenization and Padding to handle variable-length sequences.
* **Multi-class Classification:** Uses Label Encoding for efficient category management.
* **Interactive UI:** A clean, responsive chat interface built with Streamlit.
* **End-to-End Pipeline:** Covers everything from training and testing to cloud deployment.

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning:** TensorFlow / Keras
* **NLP:** Tokenizer, Padding, Scikit-learn (LabelEncoder)
* **Web Framework:** Streamlit
  
## 📂 Project Structure
```text
Aura
│
├── models/
│   ├── chat_model.h5
│   ├── label_encoder.pickle
│   └── tokenizer.pickle
│
├── notebook/
│   └── train_model.ipynb
│
├── app.py
├── LICENSE
├── README.md
└── requirements.txt
```