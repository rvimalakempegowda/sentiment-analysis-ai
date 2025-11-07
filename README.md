# Sentiment Analysis on Movie Reviews

This project demonstrates a simple natural language processing pipeline to classify movie reviews as positive or negative. It uses Python, scikit‑learn and natural language processing techniques to train a sentiment classifier on a small labelled dataset.

## Overview

- **Dataset:**  Uses the [NLTK movie reviews dataset](https://www.nltk.org/book/ch06.html) or any CSV with text and sentiment labels.
- **Preprocessing:** Tokenizes the text, removes stopwords and applies term frequency–inverse document frequency (TF‑IDF) vectorization.
- **Model:**  A `LogisticRegression` classifier from scikit‑learn.
- **Evaluation:**  Splits the data into train/test sets and reports accuracy on the test set.
- **Usage:**  Provides command‑line entry points to train the model and make predictions on new sentences.

## Requirements

- Python 3.10+
- scikit‑learn
- pandas
- nltk

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Running the project

1. **Prepare the dataset:**  Modify `sentiment_analysis.py` to point to your dataset file if using a custom dataset.
2. **Train the model:**

   ```bash
   python sentiment_analysis.py --train
   ```
3. **Evaluate and predict:**
   To evaluate the model on the test set:

   ```bash
   python sentiment_analysis.py --evaluate
   ```

   To predict the sentiment of a custom review:

   ```bash
   python sentiment_analysis.py --predict "This movie was fantastic!"
   ```

This project is meant as a starting point for anyone interested in text classification and sentiment analysis.
