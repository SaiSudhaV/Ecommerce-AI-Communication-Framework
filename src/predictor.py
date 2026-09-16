"""
Predictor module for the Ecommerce AI Communication Framework web UI.

Loads a trained model (or trains a lightweight one on first use) and exposes
a simple predict() interface for the Flask app.
"""
import os
import re
import gzip
import shutil
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

DATASET_CSV = os.path.join(DATA_DIR, "cumulative_ai_customer_communication_dataset.csv")
DATASET_GZ = DATASET_CSV + ".gz"

# New user-submitted interactions are appended here for future retraining.
FEEDBACK_CSV = os.path.join(DATA_DIR, "user_feedback_log.csv")
FEEDBACK_COLUMNS = [
    "timestamp", "customer_message", "category", "channel_name",
    "agent_shift", "tenure_bucket", "response_time_minutes", "issue_hour",
    "issue_day_of_week", "predicted_label", "predicted_prediction",
    "probability_positive", "confidence",
]

# Category escalation weights (mirrors advanced feature engineering)
CATEGORY_OPTIONS = [
    "Order Related", "Returns", "Refund Related", "Cancellation",
    "Product Queries", "Feedback", "Payments related", "Shopzilla Related",
]
CHANNEL_OPTIONS = ["Inbound", "Outcall", "Email"]
SHIFT_OPTIONS = ["Morning", "Afternoon", "Evening", "Night", "Split"]
TENURE_OPTIONS = ["On Job Training", "0-30", "31-60", "61-90", ">90"]

TENURE_MAP = {"On Job Training": 0, "0-30": 1, "31-60": 2, "61-90": 3, ">90": 4}


class SatisfactionPredictor:
    """Loads/trains a model and predicts customer satisfaction."""

    def __init__(self):
        self.model = None
        self.tfidf = None
        self.channel_map = {}
        self.category_map = {}
        self.subcategory_map = {}
        self.shift_map = {}
        self._stop_words = None
        self._lemmatizer = None
        self.structured_features = [
            "response_time_minutes", "issue_hour", "issue_day_of_week",
            "channel_encoded", "category_encoded", "subcategory_encoded",
            "shift_encoded", "tenure_encoded", "message_length", "word_count",
            "has_message", "cleaned_word_count",
        ]
        self.is_ready = False

    # ---------- text preprocessing ----------
    def _ensure_nltk(self):
        import nltk
        for res in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
            nltk.download(res, quiet=True)
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        self._stop_words = set(stopwords.words("english"))
        self._lemmatizer = WordNetLemmatizer()

    def preprocess_text(self, text):
        if pd.isna(text) or not isinstance(text, str):
            return ""
        from nltk.tokenize import word_tokenize
        text = text.lower().encode("ascii", "ignore").decode("ascii")
        text = re.sub(r"[^a-z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        tokens = word_tokenize(text)
        tokens = [self._lemmatizer.lemmatize(t) for t in tokens
                  if t not in self._stop_words and len(t) > 1]
        return " ".join(tokens)

    # ---------- data loading ----------
    def _decompress_if_needed(self):
        if not os.path.exists(DATASET_CSV) and os.path.exists(DATASET_GZ):
            with gzip.open(DATASET_GZ, "rb") as f_in, open(DATASET_CSV, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

    def load_and_train(self):
        """Train a lightweight combined model for the UI (fast, TF-IDF + structured)."""
        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder
        from scipy.sparse import hstack, csr_matrix

        self._ensure_nltk()
        self._decompress_if_needed()

        df = pd.read_csv(DATASET_CSV, low_memory=False)
        df["issue_reported_at"] = pd.to_datetime(df["issue_reported_at"], errors="coerce", dayfirst=True)
        df["issue_responded"] = pd.to_datetime(df["issue_responded"], errors="coerce", dayfirst=True)
        df["target"] = (df["csat_score"] >= 4).astype(int)
        df["cleaned_message"] = df["customer_message"].apply(self.preprocess_text)

        df["response_time_minutes"] = (
            (df["issue_responded"] - df["issue_reported_at"]).dt.total_seconds() / 60
        ).clip(lower=0).fillna(0)
        df["issue_hour"] = df["issue_reported_at"].dt.hour.fillna(0).astype(int)
        df["issue_day_of_week"] = df["issue_reported_at"].dt.dayofweek.fillna(0).astype(int)

        le_channel = LabelEncoder(); df["channel_encoded"] = le_channel.fit_transform(df["channel_name"].fillna("Unknown"))
        le_cat = LabelEncoder(); df["category_encoded"] = le_cat.fit_transform(df["category"].fillna("Unknown"))
        le_sub = LabelEncoder(); df["subcategory_encoded"] = le_sub.fit_transform(df["sub-category"].fillna("Unknown"))
        le_shift = LabelEncoder(); df["shift_encoded"] = le_shift.fit_transform(df["agent_shift"].fillna("Unknown"))
        df["tenure_encoded"] = df["tenure_bucket"].map(TENURE_MAP).fillna(0).astype(int)
        df["has_message"] = (df["cleaned_message"] != "").astype(int)
        df["cleaned_word_count"] = df["cleaned_message"].apply(lambda x: len(x.split()) if x else 0)

        # store encoders as dicts for inference
        self.channel_map = dict(zip(le_channel.classes_, le_channel.transform(le_channel.classes_)))
        self.category_map = dict(zip(le_cat.classes_, le_cat.transform(le_cat.classes_)))
        self.subcategory_map = dict(zip(le_sub.classes_, le_sub.transform(le_sub.classes_)))
        self.shift_map = dict(zip(le_shift.classes_, le_shift.transform(le_shift.classes_)))

        X_struct = df[self.structured_features].fillna(0)
        y = df["target"]
        X_tr, _, y_tr, _ = train_test_split(X_struct, y, test_size=0.2, random_state=42, stratify=y)

        self.tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5)
        X_text_tr = self.tfidf.fit_transform(df.loc[X_tr.index, "cleaned_message"])
        X_comb_tr = hstack([X_text_tr, csr_matrix(X_tr.values)])

        self.model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
        self.model.fit(X_comb_tr, y_tr)
        self.is_ready = True

    # ---------- inference ----------
    def predict(self, message, category, channel, shift, tenure,
                response_time_minutes, issue_hour=12, issue_day_of_week=2):
        """Return prediction dict for a single interaction."""
        from scipy.sparse import hstack, csr_matrix

        if not self.is_ready:
            self.load_and_train()

        cleaned = self.preprocess_text(message)
        raw_len = len(message) if isinstance(message, str) else 0
        word_count = len(message.split()) if isinstance(message, str) else 0

        row = {
            "response_time_minutes": float(response_time_minutes),
            "issue_hour": int(issue_hour),
            "issue_day_of_week": int(issue_day_of_week),
            "channel_encoded": self.channel_map.get(channel, 0),
            "category_encoded": self.category_map.get(category, 0),
            "subcategory_encoded": 0,
            "shift_encoded": self.shift_map.get(shift, 0),
            "tenure_encoded": TENURE_MAP.get(tenure, 0),
            "message_length": raw_len,
            "word_count": word_count,
            "has_message": 1 if cleaned else 0,
            "cleaned_word_count": len(cleaned.split()) if cleaned else 0,
        }
        X_struct = pd.DataFrame([row])[self.structured_features]
        X_text = self.tfidf.transform([cleaned])
        X = hstack([X_text, csr_matrix(X_struct.values)])

        prob = float(self.model.predict_proba(X)[0, 1])
        pred = int(prob >= 0.5)
        return {
            "prediction": "Satisfied (Positive)" if pred == 1 else "Dissatisfied (Negative)",
            "label": pred,
            "probability_positive": round(prob * 100, 2),
            "probability_negative": round((1 - prob) * 100, 2),
            "cleaned_message": cleaned,
            "confidence": round(max(prob, 1 - prob) * 100, 2),
        }


def log_feedback(inputs, result):
    """
    Append a user-submitted interaction and its prediction to the feedback log.

    This accumulates new prompts/contexts over time so the dataset can be
    extended and models retrained in the future. Writes to data/user_feedback_log.csv.

    Args:
        inputs (dict): the raw interaction fields submitted by the user.
        result (dict): the prediction result returned by SatisfactionPredictor.predict().
    """
    import csv
    from datetime import datetime

    os.makedirs(DATA_DIR, exist_ok=True)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "customer_message": inputs.get("message", ""),
        "category": inputs.get("category", ""),
        "channel_name": inputs.get("channel", ""),
        "agent_shift": inputs.get("shift", ""),
        "tenure_bucket": inputs.get("tenure", ""),
        "response_time_minutes": inputs.get("response_time_minutes", ""),
        "issue_hour": inputs.get("issue_hour", ""),
        "issue_day_of_week": inputs.get("issue_day_of_week", ""),
        "predicted_label": result.get("label", ""),
        "predicted_prediction": result.get("prediction", ""),
        "probability_positive": result.get("probability_positive", ""),
        "confidence": result.get("confidence", ""),
    }

    file_exists = os.path.exists(FEEDBACK_CSV)
    with open(FEEDBACK_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FEEDBACK_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    return FEEDBACK_CSV


# Singleton instance for the web app
_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = SatisfactionPredictor()
    return _predictor
