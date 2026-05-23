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
4. Implement explainable AI techniques for transparent decision-making
5. Optimize communication strategies based on predictive insights

### 1.3 Scope

The framework operates on a cumulative dataset of 85,907 customer communication records spanning multiple channels, categories, and agent interactions, combined with customer behavioral and demographic attributes.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PredictCX-NLP Pipeline                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [Raw Data] → [Data Validation] → [Missing Value Imputation]       │
│      ↓                                                              │
│  [NLP Text Preprocessing] → [Feature Engineering]                   │
│      ↓                                                              │
│  [Predictive Modeling] → [Explainable AI] → [Optimization]         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
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
| `category` | Categorical | Issue category (Returns, Order Related, etc.) | Feature |
| `sub-category` | Categorical | Issue sub-category | Feature |
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

The preprocessing pipeline addresses data quality through systematic imputation strategies:

| Data Type | Imputation Strategy | Justification |
|-----------|-------------------|---------------|
| Numerical | Median imputation | Robust to outliers; preserves central tendency |
| Categorical | Mode imputation (fallback: "Unknown") | Maintains distribution; handles fully-missing columns |
| DateTime | Forward-fill + Backward-fill | Preserves temporal ordering; ensures continuity |

**Validation**: Post-imputation assertion ensures zero missing values across all 33 columns.

### 4.2 NLP Text Preprocessing

The `customer_message` column undergoes a multi-stage NLP preprocessing pipeline:

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

Engineered features are categorized into four groups:

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

### 4.4 NLP Models (Planned/Applicable)

The framework is designed to support the following NLP model architectures:

| Model | Type | Application | Reference |
|-------|------|-------------|-----------|
| TF-IDF + Logistic Regression | Baseline | Text classification | Joachims (1998) |
| TF-IDF + SVM | Classical ML | Sentiment classification | Pang et al. (2002) |
| Word2Vec + LSTM | Deep Learning | Sequence modeling | Mikolov et al. (2013) |
| BERT (bert-base-uncased) | Transformer | Contextual embeddings | Devlin et al. (2019) |
| DistilBERT | Transformer (distilled) | Efficient inference | Sanh et al. (2019) |
| XGBoost + NLP features | Ensemble | Hybrid structured + text | Chen & Guestrin (2016) |
| SHAP + LIME | Explainable AI | Model interpretability | Lundberg & Lee (2017) |

### 4.5 Evaluation Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| Accuracy | (TP+TN) / (TP+TN+FP+FN) | Overall performance |
| Precision | TP / (TP+FP) | False positive minimization |
| Recall | TP / (TP+FN) | False negative minimization |
| F1-Score | 2×(P×R)/(P+R) | Balanced performance |
| AUC-ROC | Area under ROC curve | Discrimination ability |
| Cohen's Kappa | Agreement beyond chance | Multi-class reliability |

---

## 5. Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.10+ |
| Data Processing | pandas, NumPy | Latest |
| NLP | NLTK, Transformers (HuggingFace) | Latest |
| Visualization | Matplotlib, Seaborn, Missingno | Latest |
| Machine Learning | Scikit-learn, XGBoost | Latest |
| Deep Learning | PyTorch / TensorFlow | Latest |
| Explainable AI | SHAP, LIME | Latest |
| Testing | pytest, Hypothesis (PBT) | Latest |
| Environment | Jupyter Notebook | Latest |

---

## 6. Project Structure

```
Ecommerce-AI-Communication-Framework/
│
├── data/
│   ├── cumulative_ai_customer_communication_dataset.csv   # Raw dataset (85,907 records)
│   └── cleaned_customer_communication_dataset.csv         # Preprocessed output
│
├── notebooks/
│   └── data_preprocessing.ipynb          # Data preprocessing, NLP, and feature engineering
│
├── src/
│   └── __init__.py                       # Source package initialization
│
├── models/
│   └── (trained model artifacts)         # Serialized model files
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

## 7. Installation and Usage

### 7.1 Environment Setup

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

### 7.2 Running the Preprocessing Pipeline

```bash
# Launch Jupyter Notebook
jupyter notebook notebooks/data_preprocessing.ipynb
```

Execute all cells sequentially. The notebook will:
1. Load and profile the raw dataset
2. Detect and visualize missing values
3. Apply imputation strategies
4. Perform NLP text preprocessing
5. Engineer features for modeling
6. Export the cleaned dataset

### 7.3 Running Tests

```bash
# Run property-based tests
pytest tests/ -v
```

---

## 8. Results and Observations

### 8.1 Data Quality

- **Missing values**: Significant missingness in demographic and behavioral columns (~80% for gender, age, city, membership_type, etc.)
- **Post-imputation**: Zero missing values across all columns after pipeline execution
- **Text coverage**: Majority of records have minimal customer messages; substantive messages (>2 words) represent a subset of interactions

### 8.2 Feature Engineering Output

- **Original features**: 33 columns
- **After engineering**: 42+ columns (including one-hot encoded channels)
- **Temporal features**: Response time (mean ~170 minutes), resolution delay (mean ~18 days)
- **Engagement score**: Composite metric combining message presence, word count, CSAT, and purchase behavior

---

## 9. Research Contributions

1. **Hybrid NLP-ML Pipeline**: Integration of unstructured text processing with structured feature engineering for comprehensive customer interaction analysis
2. **Multi-strategy Imputation**: Type-aware imputation (median/mode/temporal-fill) preserving statistical properties of heterogeneous data
3. **Composite Feature Design**: Novel engagement score combining textual, behavioral, and satisfaction signals
4. **Property-Based Testing**: Formal correctness verification of preprocessing logic using Hypothesis framework
5. **Explainable Predictions**: SHAP/LIME integration for transparent model decisions in customer-facing applications

---

## 10. Future Work

- Real-time inference pipeline deployment using FastAPI/Streamlit
- Reinforcement learning for adaptive communication strategy optimization
- Multimodal analysis incorporating voice sentiment and behavioral signals
- Transfer learning with domain-adapted BERT models for e-commerce text
- A/B testing framework for communication strategy evaluation

---

## 11. References

1. Devlin, J., Chang, M.W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL-HLT*.
2. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT. *NeurIPS Workshop*.
3. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD*.
4. Lundberg, S.M., & Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
5. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space. *ICLR Workshop*.
6. Pang, B., Lee, L., & Vaithyanathan, S. (2002). Thumbs up? Sentiment Classification using Machine Learning Techniques. *EMNLP*.
7. Joachims, T. (1998). Text Categorization with Support Vector Machines. *ECML*.

---

## License

This project is developed for academic research purposes.

---

*Last updated: May 2026*
