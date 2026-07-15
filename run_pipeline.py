#!/usr/bin/env python3
"""
Ecommerce AI Communication Framework - Unified Pipeline Runner
===============================================================
This script orchestrates the complete ML pipeline, executing all stages
and producing consolidated results for README and conference presentations.

Stages:
  1. Data Preprocessing & Feature Engineering
  2. Model Training (baseline models)
  3. Hyperparameter Optimization
  4. Model Evaluation
  5. Comparative Analysis
  6. Transformer Benchmarking (optional)
  7. Explainable AI Analysis
  8. Results Consolidation & Report Generation

Usage:
    python run_pipeline.py              # Run full pipeline
    python run_pipeline.py --skip-transformers  # Skip BERT/SBERT (faster)
    python run_pipeline.py --stage evaluation   # Run specific stage only
"""

import pandas as pd
import numpy as np
import re
import warnings
import time
import os
import sys
import json
import argparse
from datetime import datetime

warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV,
    StratifiedKFold, cross_val_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from scipy.sparse import hstack, csr_matrix
from scipy.stats import ttest_rel, friedmanchisquare, randint, uniform, loguniform
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for script execution
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Optional imports
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import lime
    import lime.lime_tabular
    HAS_LIME = True
except ImportError:
    HAS_LIME = False


# ============================================================
# Configuration
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASET_PATH = os.path.join(DATA_DIR, "cumulative_ai_customer_communication_dataset.csv")
RANDOM_STATE = 42
TEST_SIZE = 0.2


def print_header(title):
    """Print formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def print_stage(stage_num, title):
    """Print stage header."""
    print(f"\n{'#'*70}")
    print(f"  STAGE {stage_num}: {title}")
    print(f"{'#'*70}\n")


# ============================================================
# Stage 1: Data Preprocessing
# ============================================================
def stage_preprocessing():
    """Load data, preprocess text, engineer features."""
    print_stage(1, "DATA PREPROCESSING & FEATURE ENGINEERING")

    # Download NLTK data
    for resource in ['punkt', 'punkt_tab', 'stopwords', 'wordnet']:
        nltk.download(resource, quiet=True)

    # Load dataset
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    print(f"Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")

    # Parse datetime
    df['issue_reported_at'] = pd.to_datetime(df['issue_reported_at'], errors='coerce', dayfirst=True)
    df['issue_responded'] = pd.to_datetime(df['issue_responded'], errors='coerce', dayfirst=True)

    # Binary target
    df['target'] = (df['csat_score'] >= 4).astype(int)
    print(f"Target: Positive={df['target'].mean()*100:.1f}%, Negative={(1-df['target'].mean())*100:.1f}%")

    # NLP Preprocessing
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):
        if pd.isna(text) or not isinstance(text, str):
            return ''
        text = text.lower()
        text = text.encode('ascii', 'ignore').decode('ascii')
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 1]
        return ' '.join(tokens)

    print("Preprocessing text...")
    df['cleaned_message'] = df['customer_message'].apply(preprocess_text)
    print(f"  Non-empty messages: {(df['cleaned_message'] != '').sum()}")

    # Structured features
    df['response_time_minutes'] = (
        (df['issue_responded'] - df['issue_reported_at']).dt.total_seconds() / 60
    ).clip(lower=0).fillna(0)
    df['issue_hour'] = df['issue_reported_at'].dt.hour.fillna(0).astype(int)
    df['issue_day_of_week'] = df['issue_reported_at'].dt.dayofweek.fillna(0).astype(int)

    for col, src in [('channel_encoded', 'channel_name'), ('category_encoded', 'category'),
                     ('subcategory_encoded', 'sub-category'), ('shift_encoded', 'agent_shift')]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[src].fillna('Unknown'))

    tenure_map = {'On Job Training': 0, '0-30': 1, '31-60': 2, '61-90': 3, '>90': 4}
    df['tenure_encoded'] = df['tenure_bucket'].map(tenure_map).fillna(0).astype(int)
    df['has_message'] = (df['cleaned_message'] != '').astype(int)
    df['cleaned_word_count'] = df['cleaned_message'].apply(lambda x: len(x.split()) if x else 0)

    structured_features = [
        'response_time_minutes', 'issue_hour', 'issue_day_of_week',
        'channel_encoded', 'category_encoded', 'subcategory_encoded',
        'shift_encoded', 'tenure_encoded', 'message_length', 'word_count',
        'has_message', 'cleaned_word_count'
    ]
    print(f"  Structured features: {len(structured_features)}")
    print("  Preprocessing complete.")
    return df, structured_features


# ============================================================
# Stage 2: Model Training
# ============================================================
def stage_training(df, structured_features):
    """Train baseline models and return predictions."""
    print_stage(2, "MODEL TRAINING")

    X_structured = df[structured_features].fillna(0)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X_structured, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5)
    X_text_train = tfidf.fit_transform(df.loc[X_train.index, 'cleaned_message'])
    X_text_test = tfidf.transform(df.loc[X_test.index, 'cleaned_message'])

    # Combined
    X_combined_train = hstack([X_text_train, csr_matrix(X_train.values)])
    X_combined_test = hstack([X_text_test, csr_matrix(X_test.values)])

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_weight = neg_count / pos_count

    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # Train models
    models = {}

    # LR Combined
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced')
    lr.fit(X_combined_train, y_train)
    models['LR (Combined)'] = {
        'model': lr, 'pred': lr.predict(X_combined_test),
        'prob': lr.predict_proba(X_combined_test)[:, 1]
    }
    print("  LR (Combined) trained.")

    # RF TF-IDF
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=RANDOM_STATE,
                                class_weight='balanced', n_jobs=-1)
    rf.fit(X_text_train, y_train)
    models['RF (TF-IDF)'] = {
        'model': rf, 'pred': rf.predict(X_text_test),
        'prob': rf.predict_proba(X_text_test)[:, 1]
    }
    print("  RF (TF-IDF) trained.")

    # XGBoost Structured
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                        scale_pos_weight=scale_weight, random_state=RANDOM_STATE,
                        eval_metric='logloss', use_label_encoder=False)
    xgb.fit(X_train, y_train)
    models['XGBoost (Struct)'] = {
        'model': xgb, 'pred': xgb.predict(X_test),
        'prob': xgb.predict_proba(X_test)[:, 1]
    }
    print("  XGBoost (Structured) trained.")

    # SVM TF-IDF
    svm = CalibratedClassifierCV(
        LinearSVC(max_iter=2000, random_state=RANDOM_STATE, class_weight='balanced'), cv=3)
    svm.fit(X_text_train, y_train)
    models['SVM (TF-IDF)'] = {
        'model': svm, 'pred': svm.predict(X_text_test),
        'prob': svm.predict_proba(X_text_test)[:, 1]
    }
    print("  SVM (TF-IDF) trained.")

    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                    random_state=RANDOM_STATE)
    gb.fit(X_train, y_train)
    models['GB (Struct)'] = {
        'model': gb, 'pred': gb.predict(X_test),
        'prob': gb.predict_proba(X_test)[:, 1]
    }
    print("  Gradient Boosting trained.")

    # Naive Bayes
    nb = MultinomialNB(alpha=1.0)
    nb.fit(X_text_train, y_train)
    models['NB (TF-IDF)'] = {
        'model': nb, 'pred': nb.predict(X_text_test),
        'prob': nb.predict_proba(X_text_test)[:, 1]
    }
    print("  Naive Bayes trained.")

    data_bundle = {
        'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
        'X_text_train': X_text_train, 'X_text_test': X_text_test,
        'X_combined_train': X_combined_train, 'X_combined_test': X_combined_test,
        'tfidf': tfidf, 'scale_weight': scale_weight
    }
    return models, data_bundle


# ============================================================
# Stage 3: Hyperparameter Optimization
# ============================================================
def stage_optimization(data_bundle):
    """Optimize model hyperparameters."""
    print_stage(3, "HYPERPARAMETER OPTIMIZATION")

    X_train = data_bundle['X_train']
    y_train = data_bundle['y_train']
    X_test = data_bundle['X_test']
    y_test = data_bundle['y_test']
    scale_weight = data_bundle['scale_weight']
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    opt_results = {}

    # GridSearchCV: XGBoost
    print("  GridSearchCV: XGBoost...")
    xgb_grid_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    start = time.time()
    xgb_grid = GridSearchCV(
        XGBClassifier(scale_pos_weight=scale_weight, random_state=RANDOM_STATE,
                      eval_metric='logloss', use_label_encoder=False),
        xgb_grid_params, cv=cv, scoring='f1', n_jobs=-1
    )
    xgb_grid.fit(X_train, y_train)
    elapsed = time.time() - start
    y_pred = xgb_grid.predict(X_test)
    opt_results['GridSearch-XGB'] = {
        'best_params': xgb_grid.best_params_,
        'cv_f1': xgb_grid.best_score_,
        'test_f1': f1_score(y_test, y_pred),
        'time': elapsed
    }
    print(f"    Best CV F1: {xgb_grid.best_score_*100:.2f}%, Time: {elapsed:.1f}s")

    # RandomizedSearchCV: XGBoost
    print("  RandomizedSearchCV: XGBoost...")
    xgb_dist = {
        'n_estimators': randint(50, 500), 'max_depth': randint(3, 12),
        'learning_rate': loguniform(0.005, 0.5), 'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4), 'min_child_weight': randint(1, 10)
    }
    start = time.time()
    xgb_random = RandomizedSearchCV(
        XGBClassifier(scale_pos_weight=scale_weight, random_state=RANDOM_STATE,
                      eval_metric='logloss', use_label_encoder=False),
        xgb_dist, n_iter=50, cv=cv, scoring='f1', n_jobs=-1, random_state=RANDOM_STATE
    )
    xgb_random.fit(X_train, y_train)
    elapsed = time.time() - start
    y_pred = xgb_random.predict(X_test)
    opt_results['RandomSearch-XGB'] = {
        'best_params': {k: (v if not hasattr(v, 'item') else v.item())
                        for k, v in xgb_random.best_params_.items()},
        'cv_f1': xgb_random.best_score_,
        'test_f1': f1_score(y_test, y_pred),
        'time': elapsed
    }
    print(f"    Best CV F1: {xgb_random.best_score_*100:.2f}%, Time: {elapsed:.1f}s")

    # Bayesian Optimization (Optuna)
    if HAS_OPTUNA:
        print("  Bayesian Optimization (Optuna): XGBoost...")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.5, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            }
            model = XGBClassifier(**params, scale_pos_weight=scale_weight,
                                  random_state=RANDOM_STATE, eval_metric='logloss',
                                  use_label_encoder=False)
            return cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1).mean()

        start = time.time()
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        study.optimize(objective, n_trials=30, show_progress_bar=False)
        elapsed = time.time() - start

        best_xgb = XGBClassifier(**study.best_params, scale_pos_weight=scale_weight,
                                 random_state=RANDOM_STATE, eval_metric='logloss',
                                 use_label_encoder=False)
        best_xgb.fit(X_train, y_train)
        y_pred = best_xgb.predict(X_test)
        opt_results['Bayesian-XGB'] = {
            'best_params': study.best_params,
            'cv_f1': study.best_value,
            'test_f1': f1_score(y_test, y_pred),
            'time': elapsed
        }
        print(f"    Best CV F1: {study.best_value*100:.2f}%, Time: {elapsed:.1f}s")
        joblib.dump(best_xgb, os.path.join(MODELS_DIR, 'optimized_xgb_bayesian.pkl'))

    # Save optimized models
    joblib.dump(xgb_grid.best_estimator_, os.path.join(MODELS_DIR, 'optimized_xgb_grid.pkl'))
    joblib.dump(xgb_random.best_estimator_, os.path.join(MODELS_DIR, 'optimized_xgb_random.pkl'))
    print("  Optimized models saved to models/")

    return opt_results


# ============================================================
# Stage 4: Model Evaluation
# ============================================================
def stage_evaluation(models, data_bundle):
    """Comprehensive model evaluation."""
    print_stage(4, "MODEL EVALUATION")

    y_test = data_bundle['y_test']
    eval_results = []

    for name, m in models.items():
        preds, probs = m['pred'], m['prob']
        eval_results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, preds),
            'Precision': precision_score(y_test, preds),
            'Recall': recall_score(y_test, preds),
            'F1-Score': f1_score(y_test, preds),
            'AUC-ROC': roc_auc_score(y_test, probs)
        })

    eval_df = pd.DataFrame(eval_results).sort_values('F1-Score', ascending=False)

    print("Performance Summary:")
    disp = eval_df.copy()
    for c in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
        disp[c] = (disp[c] * 100).round(2).astype(str) + '%'
    disp['AUC-ROC'] = disp['AUC-ROC'].round(4)
    print(disp.to_string(index=False))

    best = eval_df.iloc[0]
    print(f"\nBest Model: {best['Model']} (F1={best['F1-Score']*100:.2f}%)")

    # Save
    eval_df.to_csv(os.path.join(RESULTS_DIR, 'evaluation_results.csv'), index=False)

    # Confusion matrices
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, (name, m) in enumerate(models.items()):
        ax = axes[i // 3, i % 3]
        cm = confusion_matrix(y_test, m['pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
        ax.set_title(name, fontsize=10)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    plt.suptitle('Confusion Matrices', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ROC curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, m in models.items():
        fpr, tpr, _ = roc_curve(y_test, m['prob'])
        auc_val = roc_auc_score(y_test, m['prob'])
        axes[0].plot(fpr, tpr, lw=2, label=f'{name} ({auc_val:.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--')
    axes[0].set_title('ROC Curves')
    axes[0].set_xlabel('FPR')
    axes[0].set_ylabel('TPR')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    for name, m in models.items():
        p, r, _ = precision_recall_curve(y_test, m['prob'])
        axes[1].plot(r, p, lw=2, label=name)
    axes[1].set_title('Precision-Recall Curves')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_pr_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Plots saved to results/")

    return eval_df


# ============================================================
# Stage 5: Comparative Analysis with Statistical Tests
# ============================================================
def stage_comparative(models, data_bundle):
    """Statistical comparison of models."""
    print_stage(5, "COMPARATIVE ANALYSIS & STATISTICAL VALIDATION")

    X_train = data_bundle['X_train']
    y_train = data_bundle['y_train']
    X_combined_train = data_bundle['X_combined_train']
    scale_weight = data_bundle['scale_weight']

    # Cross-validation for statistical tests
    cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    cv_models = {
        'LR': (LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced'),
               X_combined_train),
        'XGBoost': (XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                   scale_pos_weight=scale_weight, random_state=RANDOM_STATE,
                                   eval_metric='logloss', use_label_encoder=False), X_train),
        'GB': (GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=RANDOM_STATE),
               X_train),
        'RF': (RandomForestClassifier(n_estimators=200, max_depth=20, random_state=RANDOM_STATE,
                                      class_weight='balanced', n_jobs=-1),
               data_bundle['X_text_train']),
    }

    print("Cross-validation (10-fold):")
    cv_results = {}
    for name, (model, X_cv) in cv_models.items():
        scores = cross_val_score(model, X_cv, y_train, cv=cv10, scoring='f1', n_jobs=-1)
        cv_results[name] = scores
        print(f"  {name:10s}: F1 = {scores.mean()*100:.2f}% +/- {scores.std()*100:.2f}%")

    # Friedman test
    print("\nFriedman Test (non-parametric):")
    stat, p = friedmanchisquare(*cv_results.values())
    print(f"  Chi2 = {stat:.4f}, p = {p:.4f}")
    print(f"  {'Significant difference detected (p<0.05)' if p < 0.05 else 'No significant difference'}")

    # Paired t-tests
    print("\nPaired t-tests:")
    names = list(cv_results.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            t, p_val = ttest_rel(cv_results[names[i]], cv_results[names[j]])
            sig = '*' if p_val < 0.05 else ''
            print(f"  {names[i]} vs {names[j]}: t={t:.3f}, p={p_val:.4f} {sig}")

    # CV boxplot
    fig, ax = plt.subplots(figsize=(8, 5))
    cv_df = pd.DataFrame(cv_results).melt(var_name='Model', value_name='F1')
    sns.boxplot(data=cv_df, x='Model', y='F1', palette='Set2', ax=ax)
    ax.set_title('Cross-Validation F1 Score Distribution (10-Fold)')
    ax.set_ylabel('F1 Score')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'cv_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    return cv_results


# ============================================================
# Stage 6: Transformer Benchmarking
# ============================================================
def stage_transformers(df, data_bundle):
    """Benchmark transformer-based embeddings."""
    print_stage(6, "TRANSFORMER BENCHMARKING")

    if not HAS_SBERT:
        print("  sentence-transformers not installed. Skipping.")
        print("  Install with: pip install sentence-transformers")
        return {}

    y_train = data_bundle['y_train']
    y_test = data_bundle['y_test']
    train_idx = data_bundle['X_train'].index
    test_idx = data_bundle['X_test'].index
    scale_weight = data_bundle['scale_weight']

    # Sentence-BERT
    print("  Loading Sentence-BERT (all-MiniLM-L6-v2)...")
    sbert = SentenceTransformer('all-MiniLM-L6-v2')

    print("  Encoding training texts...")
    start = time.time()
    X_sbert_train = sbert.encode(
        df.loc[train_idx, 'customer_message'].fillna('').tolist(),
        batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )
    X_sbert_test = sbert.encode(
        df.loc[test_idx, 'customer_message'].fillna('').tolist(),
        batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )
    encode_time = time.time() - start
    print(f"  Embeddings: {X_sbert_train.shape}, Time: {encode_time:.1f}s")

    # Classify with LR
    lr_sbert = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced')
    lr_sbert.fit(X_sbert_train, y_train)
    y_pred = lr_sbert.predict(X_sbert_test)
    y_prob = lr_sbert.predict_proba(X_sbert_test)[:, 1]
    lr_results = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_prob)
    }
    print(f"  LR+SBERT: Acc={lr_results['Accuracy']*100:.2f}%, F1={lr_results['F1']*100:.2f}%, AUC={lr_results['AUC']:.4f}")

    # Classify with XGBoost
    xgb_sbert = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                               scale_pos_weight=scale_weight, random_state=RANDOM_STATE,
                               eval_metric='logloss', use_label_encoder=False)
    xgb_sbert.fit(X_sbert_train, y_train)
    y_pred = xgb_sbert.predict(X_sbert_test)
    y_prob = xgb_sbert.predict_proba(X_sbert_test)[:, 1]
    xgb_results = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_prob)
    }
    print(f"  XGB+SBERT: Acc={xgb_results['Accuracy']*100:.2f}%, F1={xgb_results['F1']*100:.2f}%, AUC={xgb_results['AUC']:.4f}")

    # Save embeddings
    np.save(os.path.join(MODELS_DIR, 'sbert_train_embeddings.npy'), X_sbert_train)
    np.save(os.path.join(MODELS_DIR, 'sbert_test_embeddings.npy'), X_sbert_test)

    transformer_results = {'LR+SBERT': lr_results, 'XGB+SBERT': xgb_results}
    return transformer_results


# ============================================================
# Stage 7: Explainable AI
# ============================================================
def stage_xai(models, data_bundle, structured_features):
    """Generate model explanations."""
    print_stage(7, "EXPLAINABLE AI (XAI)")

    X_train = data_bundle['X_train']
    X_test = data_bundle['X_test']
    y_test = data_bundle['y_test']
    xgb_model = models['XGBoost (Struct)']['model']

    # Built-in feature importance
    print("  Built-in Feature Importance (XGBoost):")
    imp_df = pd.DataFrame({
        'Feature': structured_features,
        'Importance': xgb_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    for _, row in imp_df.head(5).iterrows():
        print(f"    {row['Feature']:25s}: {row['Importance']:.4f}")

    # Permutation importance
    print("\n  Permutation Importance:")
    perm_imp = permutation_importance(xgb_model, X_test, y_test,
                                      n_repeats=10, random_state=RANDOM_STATE,
                                      scoring='f1', n_jobs=-1)
    perm_df = pd.DataFrame({
        'Feature': structured_features,
        'Importance': perm_imp.importances_mean
    }).sort_values('Importance', ascending=False)
    for _, row in perm_df.head(5).iterrows():
        print(f"    {row['Feature']:25s}: {row['Importance']:.4f}")

    # SHAP
    if HAS_SHAP:
        print("\n  SHAP Analysis...")
        explainer = shap.TreeExplainer(xgb_model)
        X_explain = X_test.iloc[:300]
        shap_values = explainer.shap_values(X_explain)

        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(shap_values, X_explain, feature_names=structured_features, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'shap_summary.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("    SHAP summary plot saved.")
    else:
        print("\n  SHAP not available (pip install shap)")

    # Feature importance comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    imp_sorted = imp_df.sort_values('Importance', ascending=True)
    axes[0].barh(imp_sorted['Feature'], imp_sorted['Importance'], color='steelblue')
    axes[0].set_title('XGBoost Built-in Importance')

    perm_sorted = perm_df.sort_values('Importance', ascending=True)
    axes[1].barh(perm_sorted['Feature'], perm_sorted['Importance'], color='coral')
    axes[1].set_title('Permutation Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Feature importance plots saved.")

    return {'builtin': imp_df, 'permutation': perm_df}


# ============================================================
# Stage 8: Results Consolidation
# ============================================================
def stage_consolidate(eval_df, opt_results, cv_results, transformer_results, xai_results):
    """Generate final consolidated report."""
    print_stage(8, "RESULTS CONSOLIDATION & REPORT GENERATION")

    report = {
        'generated_at': datetime.now().isoformat(),
        'pipeline_version': '1.0.0',
        'dataset': 'cumulative_ai_customer_communication_dataset.csv',
        'target': 'csat_score >= 4 (Binary Classification)',
        'evaluation': eval_df.to_dict(orient='records'),
        'optimization': {k: {kk: (vv if not isinstance(vv, np.floating) else float(vv))
                              for kk, vv in v.items()}
                         for k, v in opt_results.items()},
        'cross_validation': {k: {'mean': float(v.mean()), 'std': float(v.std())}
                             for k, v in cv_results.items()},
        'transformer_benchmarking': transformer_results,
        'xai_top_features': xai_results['builtin'].head(5)['Feature'].tolist(),
        'available_libraries': {
            'optuna': HAS_OPTUNA,
            'sentence_transformers': HAS_SBERT,
            'shap': HAS_SHAP,
            'lime': HAS_LIME
        }
    }

    # Save JSON report
    report_path = os.path.join(RESULTS_DIR, 'pipeline_results.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Full report saved to: {report_path}")

    # Print summary for README/conference
    print_header("FINAL RESULTS SUMMARY")
    print("Model Performance (Test Set):")
    print("-" * 60)
    for _, row in eval_df.iterrows():
        print(f"  {row['Model']:20s}  F1={row['F1-Score']*100:.2f}%  AUC={row['AUC-ROC']:.4f}")

    print(f"\nBest Model: {eval_df.iloc[0]['Model']}")
    print(f"  Accuracy:  {eval_df.iloc[0]['Accuracy']*100:.2f}%")
    print(f"  Precision: {eval_df.iloc[0]['Precision']*100:.2f}%")
    print(f"  Recall:    {eval_df.iloc[0]['Recall']*100:.2f}%")
    print(f"  F1-Score:  {eval_df.iloc[0]['F1-Score']*100:.2f}%")
    print(f"  AUC-ROC:   {eval_df.iloc[0]['AUC-ROC']:.4f}")

    if opt_results:
        print(f"\nBest Optimized: {max(opt_results.items(), key=lambda x: x[1]['cv_f1'])[0]}")
        best_opt = max(opt_results.values(), key=lambda x: x['cv_f1'])
        print(f"  CV F1: {best_opt['cv_f1']*100:.2f}%")

    if transformer_results:
        print(f"\nTransformer Results:")
        for name, res in transformer_results.items():
            print(f"  {name}: F1={res['F1']*100:.2f}%, AUC={res['AUC']:.4f}")

    print(f"\nTop 5 Important Features:")
    for _, row in xai_results['builtin'].head(5).iterrows():
        print(f"  {row['Feature']}")

    print(f"\nAll results saved to: {RESULTS_DIR}/")
    print(f"All models saved to: {MODELS_DIR}/")
    return report


# ============================================================
# Main Entry Point
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Ecommerce AI Communication Framework - Pipeline Runner')
    parser.add_argument('--skip-transformers', action='store_true',
                        help='Skip transformer benchmarking (faster execution)')
    parser.add_argument('--stage', type=str, default=None,
                        choices=['preprocessing', 'training', 'optimization',
                                 'evaluation', 'comparative', 'transformers', 'xai'],
                        help='Run a specific stage only')
    args = parser.parse_args()

    total_start = time.time()
    print_header("ECOMMERCE AI COMMUNICATION FRAMEWORK")
    print(f"Pipeline started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Options: skip_transformers={args.skip_transformers}, stage={args.stage}")

    # Stage 1: Preprocessing
    df, structured_features = stage_preprocessing()
    if args.stage == 'preprocessing':
        return

    # Stage 2: Training
    models, data_bundle = stage_training(df, structured_features)
    if args.stage == 'training':
        return

    # Stage 3: Optimization
    opt_results = stage_optimization(data_bundle)
    if args.stage == 'optimization':
        return

    # Stage 4: Evaluation
    eval_df = stage_evaluation(models, data_bundle)
    if args.stage == 'evaluation':
        return

    # Stage 5: Comparative Analysis
    cv_results = stage_comparative(models, data_bundle)
    if args.stage == 'comparative':
        return

    # Stage 6: Transformer Benchmarking
    transformer_results = {}
    if not args.skip_transformers:
        transformer_results = stage_transformers(df, data_bundle)
    else:
        print_stage(6, "TRANSFORMER BENCHMARKING (SKIPPED)")
    if args.stage == 'transformers':
        return

    # Stage 7: XAI
    xai_results = stage_xai(models, data_bundle, structured_features)
    if args.stage == 'xai':
        return

    # Stage 8: Consolidation
    report = stage_consolidate(eval_df, opt_results, cv_results, transformer_results, xai_results)

    total_time = time.time() - total_start
    print(f"\nTotal pipeline time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print("Pipeline completed successfully.")


if __name__ == '__main__':
    main()
