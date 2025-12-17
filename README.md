# 🧠 Explainable Sentiment Analysis (NLP Project)
A complete NLP pipeline for **Sentiment Analysis** with **Explainable AI (XAI)** using **LIME** to interpret model predictions.  
This project demonstrates how to build, train, save, load, and explain a sentiment classifier using **TF-IDF + Logistic Regression**, with support for real-time inference.

---

## 🚀 Project Overview
Traditional machine learning models make predictions without explaining *why* a specific decision was made.  
This project solves that problem by integrating **LIME (Local Interpretable Model-Agnostic Explanations)** to highlight the words that influenced each prediction — both positively and negatively.

The dataset used is **IMDB Movie Reviews**, a widely used benchmark for sentiment analysis (positive vs negative).

---

## ✨ Key Features
- 🔤 **Text Preprocessing** (cleaning, lowercasing, punctuation removal)
- 📊 **TF-IDF Vectorization** (unigrams + bigrams)
- 🤖 **Logistic Regression Classifier**
- 💾 **Model Saving & Loading using joblib**
- 🔍 **Explainable AI using LIME**
- 📝 Inference module for testing custom user input
- 📈 High accuracy model with interpretable outputs
