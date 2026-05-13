# SMS Spam Classifier

A Machine Learning based SMS Spam Classifier built using Python, TF-IDF Vectorization, and Multinomial Naive Bayes.

The model classifies SMS messages as:
- HAM (Not Spam)
- SPAM

---

# Features

- SMS spam detection using Machine Learning
- TF-IDF text vectorization
- Multinomial Naive Bayes classifier
- Command Line Interface (CLI)
- Model saving using Pickle
- Classification report and confusion matrix evaluation

---

# Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Multinomial Naive Bayes
- Pickle

---

# Project Structure

```bash
Spam-Classifier/
│
├── train.py
├── app.py
├── spam.csv
├── spam_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Dataset

This project uses the SMS Spam Collection Dataset from Kaggle.

Dataset contains labeled SMS messages:
- ham
- spam

---

# Model Workflow

1. Load dataset
2. Clean and preprocess data
3. Convert labels into numerical values
4. Transform text using TF-IDF
5. Split dataset into training and testing sets
6. Train the model using Multinomial Naive Bayes
7. Evaluate model performance
8. Save trained model and vectorizer

---

# Model Performance

Accuracy achieved:
```bash
96%
```

Confusion Matrix:
```bash
[[966   0]
 [ 47 102]]
```

---

# Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
```

Move into the project directory:

```bash
cd Spam-Classifier
```

Create virtual environment:

```bash
python -m venv venv
```

Activate virtual environment (Windows):

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Run Training Script

```bash
python train.py
```

---

# Run CLI Application

```bash
python app.py
```

Example:

```bash
Enter a message:
WINNER! Claim your free iPhone now!
```

Output:

```bash
This message is SPAM
```

---

# Future Improvements

- GUI Interface
- Flask Web Application
- Deep Learning based classifier
- Better preprocessing techniques
- Model deployment

---

# Topics

machine-learning, python, nlp, spam-classifier, scikit-learn, tfidf, naive-bayes, text-classification
