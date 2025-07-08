# 💬 ChatInsight - Rule-Based Sentiment-Aware Chatbot

**ChatInsight** is a beginner-friendly, lightweight chatbot application built using **Streamlit**, **NLTK**, and **TextBlob**. It focuses on understanding user input through Natural Language Processing and responding appropriately based on sentiment and intent using a rule-based system.
---

## 📸 Sample UI

![Chat Screenshot](Note%20book/Screenshot%202025-07-08%20150511.png)


## 🚀 Features

- 🧠 **Intent Detection**  
  The chatbot detects user intent using simple keyword-based rules (e.g., greetings, farewells, gratitude, help requests).

- 😀 **Sentiment-Aware Responses**  
  Uses TextBlob to analyze the sentiment of the user's input and adjusts responses accordingly (positive, neutral, or negative tone).

- 🧹 **Robust Text Preprocessing**  
  Custom utility functions in `nlp_utils.py` handle:
  - Lowercasing
  - Contraction removal
  - Tokenization
  - Stopword removal
  - Lemmatization

- ⚡ **Rule-Based Reply Logic**  
  Replies are selected from a predefined response dictionary based on detected intent and sentiment score.

- 🧪 **Interactive Streamlit UI**  
  The user interface is minimal and clean, powered by Streamlit — making interaction simple and real-time.

- 🔄 **Modular Design**  
  Code is modular, with clear separation between logic (`app.py`), NLP utilities (`nlp_utils.py`), and assets (in `data/` folder).

---

