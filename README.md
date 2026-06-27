# PredictCX-NLP: AI-Driven Customer Communication Optimization Framework for E-Commerce

## Abstract

PredictCX-NLP is a hybrid artificial intelligence framework designed for intelligent customer communication management in e-commerce systems. The framework integrates Natural Language Processing (NLP), predictive machine learning models, and Explainable AI (XAI) techniques to analyze customer interactions, predict escalation or churn risks, and optimize communication strategies. The system processes unstructured customer messages through a multi-stage pipeline encompassing text preprocessing, feature engineering, and predictive modeling to deliver actionable insights for customer experience optimization.

---

## 1. Introduction

### 1.1 Problem Statement

E-commerce platforms generate large volumes of customer communication data across multiple channels (inbound calls, outbound calls, emails). Identifying high-risk interactions, predicting customer dissatisfaction, and prioritizing support cases remain significant challenges that directly impact customer retention and operational efficiency.

### 1.2 Objectives

1. Develop an NLP pipeline for processing and analyzing customer communication text data
2. Engineer domain-specific features from temporal, behavioral, and textual attributes
3. Build predictive models for customer satisfaction classification and escalation risk assessment
4. Implement intent classification for automated routing of customer queries
5. Apply Explainable AI (SHAP/LIME) techniques for transparent decision-making
6. Design a hybrid predictive communication engine combining NLP and structured ML
7. Optimize communication strategies using reinforcement learning

### 1.3 Scope

The framework operates on a cumulative dataset of 85,907 customer communication records spanning multiple channels, categories, and agent interactions, combined with customer behavioral and demographic attributes.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PredictCX-NLP Pipeline                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [Raw Data] → [Data Validation] → [Missing Value Imputation]               │
│      ↓                                                                      │
│  [NLP Text Preprocessing] → [Feature Engineering]                           │
│      ↓                                                                      │
│  [Intent Classification] → [Satisfaction Prediction]                        │
│      ↓                                                                      │
│  [Model Ensemble] → [Explainable AI (SHAP/LIME)]                           │
│      ↓                                                                      │
│  [Reinforcement Learning Optimization] → [Communication Strategy Output]   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Dataset Description

### 3.1 Data Source

- **File**: `data/cumulative_ai_customer_communication_dataset.csv`
- **Records**: 85,907 customer interactions
- **Features**: 33 original attributes
- **Channels**: Inbound, Outcall, Email

### 3.2 Data Dictionary

| Column | Data Type | Description | Role |
|--------|-----------|-------------|------|
| `unique_id` | UUID | Unique interaction identifier | Primary key |
| `channel_name` | Categorical | Communication channel (Inbound/Outcall/Email) | Feature |
| `category` | Categorical | Issue category (Returns, Order Related, etc.) | Feature / Intent label |
| `sub-category` | Categorical | Issue sub-category | Feature / Intent label |
| `customer_message` | Text | Raw customer communication text | NLP input |
| `order_id` | UUID | Associated order identifier | Linking |
| `order_date_time` | DateTime | Order placement timestamp | Temporal feature |
| `issue_reported_at` | DateTime | Issue reporting timestamp | Temporal feature |
| `issue_responded` | DateTime | Agent response timestamp | Temporal feature |
| `survey_response_date` | DateTime | Survey completion date | Temporal feature |
| `customer_city` | Categorical | Customer location (from interaction) | Feature |
| `product_category` | Categorical | Product type | Feature |
| `item_price` | Numeric | Product price (INR) | Feature |
| `connected_handling_time` | Numeric | Call handling duration | Feature |
| `agent_name` | Categorical | Support agent identifier | Feature |
| `supervisor` | Categorical | Agent supervisor | Feature |
| `manager` | Categorical | Agent manager | Feature |
| `tenure_bucket` | Ordinal | Agent experience level | Feature |
| `agent_shift` | Categorical | Agent working shift | Feature |
| `csat_score` | Numeric (1-5) | Customer satisfaction score | Target variable |
| `customer_id` | Integer | Customer identifier | Linking |
| `message_length` | Integer | Character count of message | Feature |
| `word_count` | Integer | Word count of message | Feature |
| `gender` | Categorical | Customer gender | Demographic |
| `age` | Numeric | Customer age | Demographic |
| `city` | Categorical | Customer city (demographic) | Demographic |
| `membership_type` | Categorical | Membership tier (Gold/Silver/Bronze) | Behavioral |
| `total_spend` | Numeric | Cumulative spending | Behavioral |
| `items_purchased` | Numeric | Total items bought | Behavioral |
| `average_rating` | Numeric | Average product rating given | Behavioral |
| `discount_applied` | Boolean | Whether discount was used | Behavioral |
| `days_since_last_purchase` | Numeric | Recency metric | Behavioral |
| `satisfaction_level` | Categorical | Overall satisfaction (Satisfied/Neutral/Unsatisfied) | Target variable |

---

## 4. Methodology

### 4.1 Data Preprocessing Pipeline

| Data Type | Imputation Strategy | Justification |
|-----------|-------------------|---------------|
| Numerical | Median imputation | Robust to outliers; preserves central tendency |
| Categorical | Mode imputation (fallback: "Unknown") | Maintains distribution; handles fully-missing columns |
| DateTime | Forward-fill + Backward-fill | Preserves temporal ordering; ensures continuity |

**Validation**: Post-imputation assertion ensures zero missing values across all 33 columns.

### 4.2 NLP Text Preprocessing

| Step | Technique | Implementation |
|------|-----------|----------------|
| 1. Case normalization | Lowercasing | `str.lower()` |
| 2. Character filtering | ASCII encoding | Remove emojis and non-ASCII characters |
| 3. Special character removal | Regex substitution | Retain only alphabetic characters and spaces |
| 4. Whitespace normalization | Regex substitution | Collapse multiple spaces |
| 5. Tokenization | Word tokenization | NLTK `word_tokenize()` |
| 6. Stopword removal | English stopword list | NLTK stopwords corpus |
| 7. Lemmatization | WordNet lemmatizer | NLTK `WordNetLemmatizer()` |

**Output**: `cleaned_message` column containing preprocessed tokens suitable for vectorization and model input.

### 4.3 Feature Engineering

#### 4.3.1 Temporal Features

| Feature | Formula | Description |
|---------|---------|-------------|
| `response_time_minutes` | `(issue_responded - issue_reported_at) / 60` | Agent response latency |
| `resolution_delay_days` | `(issue_reported_at - order_date_time) / 86400` | Time from order to issue |
| `issue_hour` | `issue_reported_at.hour` | Hour of day (0-23) |
| `issue_day_of_week` | `issue_reported_at.dayofweek` | Day of week (0=Mon, 6=Sun) |

#### 4.3.2 Text-Derived Features

| Feature | Formula | Description |
|---------|---------|-------------|
| `cleaned_message` | NLP pipeline output | Preprocessed text for modeling |
| `cleaned_word_count` | `len(cleaned_message.split())` | Token count after preprocessing |
| `cleaned_message_length` | `len(cleaned_message)` | Character length after preprocessing |
| `has_message` | `cleaned_word_count > 2` | Binary: meaningful message indicator |

#### 4.3.3 Encoding Features

| Feature | Encoding Type | Mapping |
|---------|--------------|---------|
| `satisfaction_encoded` | Ordinal | Unsatisfied=0, Neutral=1, Satisfied=2 |
| `shift_encoded` | Ordinal | Morning=0, Afternoon=1, Evening=2, Night=3, Split=4 |
| `tenure_encoded` | Ordinal | On Job Training=0, 0-30=1, 31-60=2, 61-90=3, >90=4 |
| `category_encoded` | Label encoding | Unique integer per category |
| `channel_*` | One-hot encoding | Binary columns per channel |

#### 4.3.4 Interaction Features

| Feature | Formula | Description |
|---------|---------|-------------|
| `spend_per_item` | `total_spend / items_purchased` | Average transaction value |
| `engagement_score` | `0.3×has_message + 0.2×wc_norm + 0.3×csat_norm + 0.2×items_norm` | Composite engagement metric |

### 4.4 Intent Classification

Customer messages are classified into intent categories derived from the `category` and `sub-category` fields:

| Intent Label | Category Source | Examples |
|--------------|----------------|----------|
| `order_status` | Order Related | Order tracking, delayed delivery |
| `return_request` | Returns | Return request, reverse pickup, exchange |
| `cancellation` | Cancellation | Order cancellation, not needed |
| `refund_inquiry` | Refund Related | Refund status, payment issues |
| `product_query` | Product Queries | Product info, life insurance |
| `feedback` | Feedback | Professional/unprofessional behaviour |
| `platform_issue` | Shopzilla Related | Signup issues, general enquiry |
| `payment_issue` | Payments related | Online payment, payment queries |

The intent classifier uses the `cleaned_message` text with TF-IDF vectorization to predict the appropriate `category` label, enabling automated routing of customer queries to specialized support teams.

---

## 5. Models and Experimental Results

### 5.1 Model Architecture Summary

The framework implements six predictive models across three paradigms:

| # | Model | Type | Features | Hyperparameters |
|---|-------|------|----------|-----------------|
| 1 | TF-IDF + Logistic Regression | NLP Baseline | TF-IDF (5000 features, bigrams) | max_iter=1000, class_weight=balanced |
| 2 | TF-IDF + Random Forest | NLP Ensemble | TF-IDF (5000 features, bigrams) | n_estimators=200, max_depth=20, class_weight=balanced |
| 3 | XGBoost | Gradient Boosting | 12 structured features | n_estimators=300, max_depth=6, lr=0.1, scale_pos_weight |
| 4 | Combined (NLP + Structured) | Hybrid LR | TF-IDF + 12 structured (5012 total) | max_iter=1000, class_weight=balanced |
| 5 | Logistic Regression (Structured) | Classical ML | 12 structured features | max_iter=1000, class_weight=balanced |
| 6 | SVM (Linear) | Classical ML | TF-IDF (5000 features, bigrams) | C=1.0, max_iter=2000, class_weight=balanced, CalibratedCV |

### 5.2 Experimental Setup

- **Target variable**: `csat_score` binarized — Positive (4-5) vs Negative (1-3)
- **Class distribution**: 82.5% Positive, 17.5% Negative (imbalanced)
- **Train/Test split**: 80/20 stratified random split (68,725 / 17,182 samples)
- **Text vectorization**: TF-IDF with max_features=5000, ngram_range=(1,2), min_df=5
- **Class balancing**: `class_weight='balanced'` or `scale_pos_weight` for all models

### 5.3 Model Comparison Results

| # | Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|-------|----------|-----------|--------|----------|---------|
| 1 | TF-IDF + Logistic Regression | 83.59% | 87.14% | 93.97% | 90.43% | 0.6992 |
| 2 | TF-IDF + Random Forest | 82.66% | 86.92% | 92.95% | 89.84% | 0.6932 |
| 3 | XGBoost (Structured) | 69.71% | 89.98% | 71.19% | 79.49% | **0.7311** |
| 4 | Combined (NLP + Structured) | 74.61% | 88.85% | 79.14% | 83.71% | 0.7273 |
| 5 | Logistic Regression (Structured) | 69.04% | 87.05% | 73.38% | 79.63% | 0.6555 |
| 6 | **SVM (TF-IDF)** | **85.10%** | 85.51% | **98.65%** | **91.61%** | 0.6911 |

### 5.4 Confusion Matrix Analysis

**Best Model: SVM (TF-IDF) — Confusion Matrix**

```
                 Predicted
              Negative  Positive
Actual  Negative   1368     1646
        Positive    191    13977
```

- **True Positives (TP)**: 13,977 — correctly identified satisfied customers
- **True Negatives (TN)**: 1,368 — correctly identified dissatisfied customers
- **False Positives (FP)**: 1,646 — dissatisfied customers misclassified as satisfied
- **False Negatives (FN)**: 191 — satisfied customers misclassified as dissatisfied

### 5.5 Precision, Recall, and F1-Score Analysis

| Model | Precision | Recall | F1-Score | Interpretation |
|-------|-----------|--------|----------|----------------|
| SVM (TF-IDF) | 85.51% | 98.65% | 91.61% | High recall: captures nearly all positive cases; good for minimizing missed satisfaction |
| TF-IDF + LR | 87.14% | 93.97% | 90.43% | Balanced precision-recall; reliable baseline |
| XGBoost | 89.98% | 71.19% | 79.49% | High precision: fewer false positives; conservative predictor |
| LR (Structured) | 87.05% | 73.38% | 79.63% | Similar to XGBoost; structured features alone are insufficient |

**Key Insight**: Text-based models (SVM, LR) achieve significantly higher recall and F1 than structured-only models, confirming that customer message content is the dominant predictive signal.

### 5.6 Fine-Tuning: Logistic Regression & SVM

#### Logistic Regression Tuning

| Parameter | Tested Values | Optimal |
|-----------|---------------|---------|
| `C` (regularization) | 0.01, 0.1, 1.0, 10.0 | 1.0 |
| `penalty` | L1, L2 | L2 |
| `solver` | liblinear, lbfgs, saga | lbfgs |
| `class_weight` | None, balanced | balanced |
| `max_iter` | 500, 1000, 2000 | 1000 |

#### SVM (LinearSVC) Tuning

| Parameter | Tested Values | Optimal |
|-----------|---------------|---------|
| `C` (regularization) | 0.01, 0.1, 1.0, 10.0 | 1.0 |
| `loss` | hinge, squared_hinge | squared_hinge |
| `class_weight` | None, balanced | balanced |
| `max_iter` | 1000, 2000, 5000 | 2000 |
| Calibration | CalibratedClassifierCV, cv=3 | Applied |

---

## 6. Advanced Models and Techniques

### 6.1 Transformer Models (BERT)

The framework is designed to incorporate BERT-based transformer models for superior contextual understanding:

| Component | Configuration |
|-----------|---------------|
| **Base Model** | `bert-base-uncased` (110M parameters) |
| **Tokenizer** | WordPiece tokenization, max_length=128 |
| **Fine-tuning** | Classification head on [CLS] token |
| **Optimizer** | AdamW, learning_rate=2e-5, weight_decay=0.01 |
| **Epochs** | 3-5 with early stopping |
| **Batch size** | 16 (gradient accumulation for larger effective batch) |
| **Alternative** | DistilBERT (66M params) for 60% faster inference |

**Expected improvement**: BERT captures contextual nuance (e.g., sarcasm, implicit dissatisfaction) that TF-IDF misses, targeting +3-5% F1 improvement over SVM baseline.

### 6.2 Explainable AI (SHAP / LIME)

| Technique | Application | Output |
|-----------|-------------|--------|
| **SHAP (SHapley Additive exPlanations)** | Global + local feature importance for XGBoost and ensemble models | SHAP summary plots, force plots, dependence plots |
| **LIME (Local Interpretable Model-agnostic Explanations)** | Per-prediction explanations for text classification | Highlighted words contributing to positive/negative predictions |
| **Feature Importance Ranking** | XGBoost built-in | Bar chart of top-12 structured features |

**Use cases**:
- Explain why a customer interaction is flagged as high-risk
- Identify which words in customer messages drive negative predictions
- Validate model decisions for regulatory compliance

### 6.3 Reinforcement Learning Optimization

The communication optimization engine uses RL to learn optimal response strategies:

| Component | Design |
|-----------|--------|
| **State** | Customer profile (membership, spend, recency) + interaction features (category, sentiment, response time) |
| **Action space** | Priority level (1-5), channel routing (escalate/retain/transfer), response template selection |
| **Reward signal** | CSAT score improvement, resolution time reduction, escalation avoidance |
| **Algorithm** | Deep Q-Network (DQN) or Proximal Policy Optimization (PPO) |
| **Training** | Offline from historical interaction-outcome pairs |

**Objective**: Maximize long-term customer satisfaction by adaptively selecting communication strategies based on real-time interaction context.

### 6.4 Hybrid Predictive Communication Engine

The hybrid engine combines all model outputs into a unified decision system:

```
┌──────────────────────────────────────────────────────────────────┐
│                  Hybrid Predictive Engine                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [NLP Model (SVM/BERT)]  →  Satisfaction Prediction              │
│  [Intent Classifier]     →  Category / Routing Decision          │
│  [XGBoost Structured]    →  Escalation Risk Score                │
│  [RL Agent]              →  Optimal Response Strategy             │
│                                                                  │
│  ─── Fusion Layer ───                                            │
│  Weighted ensemble + business rules → Final recommendation       │
│                                                                  │
│  [SHAP/LIME]             →  Explanation for human review          │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Decision outputs**:
1. **Satisfaction prediction**: Positive/Negative with confidence score
2. **Intent classification**: Automated category assignment for routing
3. **Escalation risk**: Probability score (0-1) for supervisor intervention
4. **Recommended action**: Priority level, suggested response template, channel routing
5. **Explanation**: SHAP/LIME-based reasoning for transparency

---

## 7. Evaluation Metrics

### 7.1 Classification Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| Accuracy | (TP+TN) / (TP+TN+FP+FN) | Overall performance |
| Precision | TP / (TP+FP) | Minimize false alarms |
| Recall (Sensitivity) | TP / (TP+FN) | Minimize missed detections |
| F1-Score | 2×(P×R)/(P+R) | Balanced precision-recall |
| AUC-ROC | Area under ROC curve | Threshold-independent discrimination |
| Cohen's Kappa | Agreement beyond chance | Multi-class reliability |
| Specificity | TN / (TN+FP) | True negative rate |

### 7.2 Model Selection Criteria

| Priority | Metric | Rationale |
|----------|--------|-----------|
| 1 | F1-Score | Best single metric for imbalanced classification |
| 2 | AUC-ROC | Threshold-independent; useful for risk scoring |
| 3 | Recall | Critical for catching dissatisfied customers |
| 4 | Precision | Important for operational cost (false escalations) |

---

## 8. Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.10+ |
| Data Processing | pandas, NumPy | Latest |
| NLP | NLTK, Transformers (HuggingFace) | Latest |
| Visualization | Matplotlib, Seaborn, Missingno | Latest |
| Machine Learning | Scikit-learn, XGBoost | Latest |
| Deep Learning | PyTorch / TensorFlow | Latest |
| Explainable AI | SHAP, LIME | Latest |
| Reinforcement Learning | Stable-Baselines3, Gymnasium | Latest |
| Testing | pytest, Hypothesis (PBT) | Latest |
| Environment | Jupyter Notebook | Latest |

---

## 9. Project Structure

```
Ecommerce-AI-Communication-Framework/
│
├── data/
│   ├── cumulative_ai_customer_communication_dataset.csv   # Raw dataset (85,907 records)
│   └── cleaned_customer_communication_dataset.csv         # Preprocessed output
│
├── notebooks/
│   ├── data_preprocessing.ipynb          # Data preprocessing, NLP, and feature engineering
│   └── model_training.ipynb              # Model training, evaluation, and comparison
│
├── src/
│   └── __init__.py                       # Source package initialization
│
├── models/
│   ├── xgboost_csat_model.pkl            # Trained XGBoost model
│   ├── combined_lr_model.pkl             # Combined LR model
│   └── tfidf_vectorizer.pkl              # Fitted TF-IDF vectorizer
│
├── tests/
│   ├── __init__.py
│   └── test_preprocessing_properties.py  # Property-based tests (Hypothesis)
│
├── evaluation/
│   └── MT24AAC019_Project_Evaluation1.pptx  # Project evaluation presentation
│
├── dashboard/
│   └── (visualization dashboard)         # Interactive dashboard components
│
├── docs/
│   └── (documentation)                   # Additional documentation
│
├── requirements.txt                      # Python dependencies
└── README.md                             # Project documentation (this file)
```

---

## 10. Installation and Usage

### 10.1 Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd Ecommerce-AI-Communication-Framework

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 10.2 Running the Preprocessing Pipeline

```bash
jupyter notebook notebooks/data_preprocessing.ipynb
```

### 10.3 Running Model Training

```bash
jupyter notebook notebooks/model_training.ipynb
```

### 10.4 Running Tests

```bash
pytest tests/ -v
```

---

## 11. Results Summary

### 11.1 Key Findings

1. **SVM achieves highest F1-score (91.61%)** with near-perfect recall (98.65%), making it optimal for detecting satisfied vs dissatisfied customers
2. **Text features dominate structured features**: NLP-based models outperform structured-only models by 10-15% in accuracy
3. **Response time is the strongest structured predictor** (correlation: -0.148 with CSAT)
4. **Class imbalance** (82.5% positive) requires careful handling via `class_weight='balanced'`
5. **Only 33.5% of records have customer messages** — models must handle sparse text gracefully

### 11.2 Best Model Performance

```
Model: SVM (Linear) with TF-IDF Features
├── Accuracy:  85.10%
├── Precision: 85.51%
├── Recall:    98.65%
├── F1-Score:  91.61%
└── AUC-ROC:   0.6911
```

---

## 12. Research Contributions

1. **Hybrid NLP-ML Pipeline**: Integration of unstructured text processing with structured feature engineering for comprehensive customer interaction analysis
2. **Multi-strategy Imputation**: Type-aware imputation (median/mode/temporal-fill) preserving statistical properties of heterogeneous data
3. **Comparative Model Analysis**: Systematic evaluation of 6 models spanning classical ML, NLP, ensemble, and hybrid approaches
4. **Intent Classification Framework**: Automated customer query categorization enabling intelligent routing
5. **Hybrid Predictive Engine Design**: Architecture combining NLP, structured ML, RL, and XAI for end-to-end communication optimization
6. **Property-Based Testing**: Formal correctness verification of preprocessing logic using Hypothesis framework
7. **Explainable Predictions**: SHAP/LIME integration for transparent model decisions in customer-facing applications

---

## 13. Future Work

1. **BERT Fine-tuning**: Domain-adapted transformer for e-commerce customer text (+3-5% expected F1 gain)
2. **Reinforcement Learning Deployment**: Train DQN/PPO agent on historical interaction-outcome pairs
3. **Real-time Inference Pipeline**: FastAPI/Streamlit deployment for live prediction
4. **Multimodal AI**: Incorporate voice sentiment and behavioral click-stream signals
5. **A/B Testing Framework**: Statistical evaluation of communication strategy improvements
6. **Active Learning**: Human-in-the-loop annotation for improving intent classification accuracy

---

## 14. References

1. Devlin, J., Chang, M.W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL-HLT*.
2. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT. *NeurIPS Workshop*.
3. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD*.
4. Lundberg, S.M., & Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
5. Ribeiro, M.T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. *KDD*.
6. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space. *ICLR Workshop*.
7. Pang, B., Lee, L., & Vaithyanathan, S. (2002). Thumbs up? Sentiment Classification using Machine Learning Techniques. *EMNLP*.
8. Joachims, T. (1998). Text Categorization with Support Vector Machines. *ECML*.
9. Sutton, R.S., & Barto, A.G. (2018). Reinforcement Learning: An Introduction (2nd ed.). *MIT Press*.
10. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.

---

## License

This project is developed for academic research purposes.

---

*Last updated: June 2026*
