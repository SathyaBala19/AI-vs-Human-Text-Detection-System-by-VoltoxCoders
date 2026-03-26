# AI-vs-Human-Text-Detection-System-by-VoltoxCoders

# 🧠 AI vs Human Text Detection System

A Machine Learning project that classifies whether a given text is **Human-written** or **AI-generated**, enhanced with a confidence-based decision layer.

---

## 🚀 Features

* ✅ Text classification (Human vs AI)
* 🔤 Text preprocessing pipeline
* 🔢 Feature extraction using CountVectorizer
* 🤖 Multiple ML models (Logistic Regression, Naive Bayes)
* 📊 Model evaluation (accuracy & metrics)
* 🧠 Intelligent decision layer based on confidence

---

## 🎯 Objective

To build a system that:

* Accurately classifies text origin (Human / AI)
* Provides **confidence-aware decisions**
* Improves interpretability using rule-based logic

---

## 📊 Dataset

### Source

* Public datasets (e.g., Kaggle AI vs Human text datasets)
* OpenAI-generated samples
* Other NLP benchmark datasets

### Details

* Labels:

  * `0 → Human-written`
  * `1 → AI-generated`
* Balanced dataset recommended
* Typical size: 5,000+ samples

---

## ⚙️ Preprocessing

Text data is cleaned using:

* Lowercasing
* Removing punctuation
* Removing special characters
* Removing extra spaces

> Optional: Stopword removal, stemming, lemmatization

---

## 🔢 Feature Extraction

We use:

```python
CountVectorizer
```

* Converts text into numerical vectors
* Based on word frequency (Bag-of-Words)

---

## 🤖 Models Used

### 1. Logistic Regression

* Linear classification model
* Provides probability scores

### 2. Naive Bayes (MultinomialNB)

* Probabilistic model
* Efficient for text classification

---

## 📈 Evaluation Metrics

* Accuracy
* Precision (optional)
* Recall (optional)
* F1-score (optional)

---

## 🧠 Intelligent Decision Layer

A rule-based system is applied on model confidence:

| Confidence  | Decision                          |
| ----------- | --------------------------------- |
| ≥ 0.80      | ✅ Acceptable (High certainty)     |
| 0.60 – 0.79 | ❓ Needs Review                    |
| < 0.60      | ⚠ Likely AI-generated / Uncertain |

---

## ⚖️ Threshold Justification

* ≥ 0.80 → High confidence → reliable prediction
* 0.60–0.80 → Moderate → requires review
* < 0.60 → Low confidence → uncertain or AI-like

Thresholds can be adjusted based on model performance.

---

## 🔄 Workflow

```text
1. Load dataset
2. Preprocess text
3. Convert to numerical features
4. Train ML models
5. Evaluate models
6. Predict new text
7. Apply decision layer
```

---

## 📂 Project Structure

```bash
AI-vs-Human-Text-Detection/
│
├── data/                # Dataset files
├── notebooks/           # Jupyter notebooks
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── decision_layer.py
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Requirements

```bash
pip install -r requirements.txt
```

Common libraries:

* scikit-learn
* pandas
* numpy

---

## 🔮 Future Enhancements

* Use TF-IDF / BERT embeddings
* Deep learning models (LSTM, Transformers)
* Web app deployment (Flask/Streamlit)
* Explainability (SHAP/LIME)

---

## 📌 Conclusion

This project demonstrates how traditional ML models combined with a simple decision layer can effectively classify and interpret AI vs human-generated text.

---
