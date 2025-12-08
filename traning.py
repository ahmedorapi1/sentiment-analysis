import re
import string
import joblib
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


def clean_text(text: str) -> str:
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # Remove mentions and hashtags
    text = re.sub(r"[@#]\w+", " ", text)
    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_imdb_data():
    print("[INFO] Loading IMDB dataset ...")
    dataset = load_dataset("imdb")

    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]

    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]

    print("[INFO] Cleaning text ...")
    train_texts = [clean_text(t) for t in train_texts]
    test_texts = [clean_text(t) for t in test_texts]

    X_train, X_val, y_train, y_val = train_test_split(
        train_texts,
        train_labels,
        test_size=0.15,
        random_state=42,
        stratify=train_labels
    )

    return X_train, X_val, test_texts, y_train, y_val, test_labels


def build_model():
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=30000,
                    ngram_range=(1, 2),
                    stop_words="english",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    n_jobs=-1,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    return model


def train_and_evaluate():
    X_train, X_val, X_test, y_train, y_val, y_test = load_imdb_data()

    model = build_model()

    print("[INFO] Training model ...")
    model.fit(X_train, y_train)

    print("[INFO] Evaluating on validation set ...")
    val_preds = model.predict(X_val)
    print("Validation accuracy:", accuracy_score(y_val, val_preds))
    print(classification_report(y_val, val_preds, digits=4))

    print("[INFO] Evaluating on test set ...")
    test_preds = model.predict(X_test)
    print("Test accuracy:", accuracy_score(y_test, test_preds))
    print(classification_report(y_test, test_preds, digits=4))

    return model, (X_test, y_test)


if __name__ == "__main__":
    model, (X_test, y_test) = train_and_evaluate()

    joblib.dump(model, "sentiment_model.pkl")

