#  Spam Email Detection using Machine Learning

##  Project Overview

This project builds a **Spam Email Classifier** using Machine Learning.
It classifies emails/messages as:

* **Ham (Not Spam)**
* **Spam**

The model is trained using **TF-IDF Vectorization** and a **Naive Bayes classifier**.

---

##  Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib & Seaborn

---

##  Dataset

* File: `email.csv`
* Columns:

  * `Category` → ham / spam
  * `Message` → email text

---

##  How It Works

1. Data cleaning and preprocessing
2. Convert text into numerical features using TF-IDF
3. Train model using Multinomial Naive Bayes
4. Evaluate performance using accuracy and confusion matrix

---

##  Model Performance

* Accuracy: **~(your accuracy here, e.g., 98%)**
* Evaluation metrics:

  * Precision
  * Recall
  * F1-score

---

##  Saved Files

* `spam_model.pkl` → trained model
* `tfidf_vectorizer.pkl` → text vectorizer

 Both files are required for predictions.

---

##  How to Use

```python
import joblib

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

text = ["Free money!!! Click here now"]
text_vec = vectorizer.transform(text)

prediction = model.predict(text_vec)

print("Spam" if prediction[0] == 1 else "Ham")
```

---

##  Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

---

##  Future Improvements

* Use deep learning (LSTM / BERT)
* Deploy as web app (Streamlit / Flask)
* Improve dataset size

---

##  Author

Hifsa Shahid

---
