# Ecommerce AI Communication Framework

A comprehensive machine learning framework for predicting customer satisfaction in e-commerce communication channels. This project implements end-to-end NLP and ML pipelines including data preprocessing, model training, hyperparameter optimization, transformer-based benchmarking, explainable AI, and comparative analysis.

## Project Structure

```
Ecommerce-AI-Communication-Framework/
├── data/
│   ├── cumulative_ai_customer_communication_dataset.csv
│   └── user_feedback_log.csv             # Auto-logged live predictions (for retraining)
├── notebooks/
│   ├── data_preprocessing.ipynb          # Data cleaning & feature engineering
│   ├── model_training.ipynb              # Baseline model training (6 models)
│   ├── advanced_feature_engineering.ipynb # CRI, sentiment, fusion features
│   ├── model_optimization.ipynb          # Hyperparameter tuning
│   ├── model_evaluation.ipynb            # Performance evaluation
│   ├── comparative_analysis.ipynb        # Statistical model comparison
│   ├── visualization.ipynb               # Results visualization
│   ├── transformer_benchmarking.ipynb    # BERT/Sentence-BERT integration
│   └── explainable_ai.ipynb              # SHAP, LIME, feature importance
├── models/                               # Saved trained models & artifacts
├── results/                              # Pipeline output (plots, CSV, JSON)
├── src/
│   ├── __init__.py
│   ├── predictor.py                      # Model loading & prediction module
│   └── support_bot.py                    # Customer support solution generator
├── templates/                            # Web UI HTML templates
│   ├── base.html                         # Shared layout
│   ├── index.html                        # Dashboard (results tables)
│   ├── support.html                      # Customer support chatbot
│   ├── predict.html                      # Live prediction interface
│   └── visualizations.html               # Results image gallery
├── static/
│   └── css/style.css                     # Web UI styling
├── evaluation/                           # Project evaluation presentations
├── app.py                                # Flask web application
├── run_pipeline.py                       # Unified pipeline script
├── requirements.txt
└── README.md
```

## Methodology

### Target Variable
- **Binary Classification**: `csat_score >= 4` (Positive) vs `csat_score < 4` (Negative)
- Customer satisfaction prediction from communication data

### Pipeline Stages

| Stage | Notebook | Description |
|-------|----------|-------------|
| 1 | `data_preprocessing.ipynb` | Data profiling, missing value imputation, NLP text cleaning, feature engineering |
| 2 | `model_training.ipynb` | Train 6 baseline models (LR, RF, XGBoost, SVM, GB, NB) |
| 3 | `model_optimization.ipynb` | GridSearchCV, RandomizedSearchCV, Bayesian optimization (Optuna) |
| 4 | `model_evaluation.ipynb` | Metrics, confusion matrices, ROC/PR curves, CV stability |
| 5 | `comparative_analysis.ipynb` | 11 model variants, Friedman test, paired t-tests |
| 6 | `transformer_benchmarking.ipynb` | Sentence-BERT & BERT embeddings vs TF-IDF baseline |
| 7 | `explainable_ai.ipynb` | SHAP, LIME, permutation importance, partial dependence |
| 8 | `visualization.ipynb` | Heatmaps, learning curves, error analysis, calibration |

### Models Implemented

- **Logistic Regression** (TF-IDF, Structured, Combined features)
- **Random Forest** (TF-IDF features)
- **XGBoost** (Structured features)
- **Support Vector Machine** (LinearSVC + Calibration)
- **Gradient Boosting** (Structured features)
- **Naive Bayes** (TF-IDF features)
- **Decision Tree** (Structured features)
- **Sentence-BERT + LR/XGBoost** (Transformer embeddings)
- **BERT (CLS token) + LR** (Contextual embeddings)

### Feature Representations
1. **TF-IDF** (5000 features, unigrams + bigrams)
2. **Structured Features** (12 engineered features: response time, hour, day, channel, category, etc.)
3. **Combined** (TF-IDF + Structured)
4. **Sentence-BERT Embeddings** (384-dim dense vectors)
5. **BERT CLS Embeddings** (768-dim contextual vectors)

### Explainability Methods
- SHAP (TreeExplainer) - global and local explanations
- LIME - local interpretable explanations
- Permutation Importance - model-agnostic feature ranking
- Partial Dependence Plots - feature effect visualization
- Logistic Regression coefficients - linear interpretability

## Installation & Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd Ecommerce-AI-Communication-Framework

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Decompress data and model files
gunzip data/*.gz models/*.gz

# 5. Download required NLTK data (auto-downloads on first run, or manually):
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Optional Dependencies (for advanced features)
```bash
# Transformer models (required for BERT/Sentence-BERT benchmarking)
pip install sentence-transformers transformers torch

# Explainable AI (required for SHAP/LIME analysis)
pip install shap lime

# Bayesian optimization (required for Optuna-based tuning)
pip install optuna
```

## How to Run

### Option 1: Run Individual Notebooks (Recommended for Exploration)

Each notebook is self-contained and can be run independently. Start Jupyter:
```bash
jupyter notebook
```

Then open notebooks in the recommended order:

| Step | Notebook | What It Does |
|------|----------|--------------|
| 1 | `notebooks/data_preprocessing.ipynb` | Cleans data, imputes missing values, engineers features, exports cleaned CSV |
| 2 | `notebooks/model_training.ipynb` | Trains 6 baseline models, shows initial performance comparison |
| 3 | `notebooks/model_optimization.ipynb` | Tunes hyperparameters using GridSearch, RandomSearch, and Bayesian optimization |
| 4 | `notebooks/model_evaluation.ipynb` | Generates confusion matrices, ROC/PR curves, cross-validation stability |
| 5 | `notebooks/comparative_analysis.ipynb` | Compares 11 model variants with statistical significance tests |
| 6 | `notebooks/transformer_benchmarking.ipynb` | Benchmarks Sentence-BERT and BERT against TF-IDF baseline |
| 7 | `notebooks/explainable_ai.ipynb` | SHAP, LIME, permutation importance, partial dependence plots |
| 8 | `notebooks/visualization.ipynb` | Heatmaps, learning curves, error analysis, calibration plots |

Each notebook runs independently — open it and execute all cells (Kernel > Run All).

### Option 2: Run Full Pipeline (Automated End-to-End)

```bash
# Run complete pipeline (all 8 stages)
python run_pipeline.py

# Run without transformer models (much faster, skips BERT/SBERT)
python run_pipeline.py --skip-transformers

# Run a specific stage only
python run_pipeline.py --stage preprocessing
python run_pipeline.py --stage training
python run_pipeline.py --stage optimization
python run_pipeline.py --stage evaluation
python run_pipeline.py --stage comparative
python run_pipeline.py --stage transformers
python run_pipeline.py --stage xai
```

### Option 3: Run from VS Code / Kiro

1. Open any `.ipynb` file in the editor
2. Select the Python interpreter from `.venv`
3. Click "Run All" or execute cells one by one

### Option 4: Web Application / Dashboard

A Flask web interface lets you explore results and run live predictions in the browser.

```bash
# 1. Install Flask (included in requirements.txt)
pip install flask

# 2. Start the web server
python app.py

# 3. Open in your browser
#    http://127.0.0.1:5001
#    (override the port with: PORT=8080 python app.py)
```

> **Troubleshooting**: The app runs on port **5001** by default because macOS uses port 5000 for AirPlay Receiver. If a prediction returns an error, make sure no stale server is running: `lsof -ti:5001 | xargs kill -9`, then restart with `python app.py`.

The web UI provides four pages:

| Page | URL | Description |
|------|-----|-------------|
| **Dashboard** | `/` | Model comparison, optimization, transformer benchmarking, cross-validation, and top XAI features — all in interactive tables |
| **Support Assistant** | `/support` | Customer-facing chatbot — enter a query and issue category, get an instant solution plus predicted sentiment |
| **Live Prediction** | `/predict` | Enter a customer message + interaction details and get a real-time satisfaction prediction with confidence scores |
| **Visualizations** | `/visualizations` | Gallery of all generated charts (confusion matrices, ROC curves, SHAP, feature importance) |

**REST API endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Support bot. Body: `{query, category}` → returns a solution + satisfaction prediction |
| `/api/predict` | POST | JSON prediction. Body: `{message, category, channel, shift, tenure, response_time_minutes, issue_hour}` |
| `/api/results` | GET | Returns full pipeline results as JSON |
| `/health` | GET | Health check |

Example API call:
```bash
curl -X POST http://127.0.0.1:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{"message":"Order arrived late and damaged","category":"Returns","channel":"Inbound","shift":"Morning","tenure":"0-30","response_time_minutes":120}'
```

> **Note**: On the first prediction, the app trains a lightweight model (~30s). Subsequent predictions are instant. The dashboard reads from `results/pipeline_results.json`, so run `python run_pipeline.py` first to populate it.

## User Guide

This section explains how to use the web application step by step.

### Getting Started

1. **Install dependencies** — `pip install -r requirements.txt`
2. **Decompress data/models** — `gunzip data/*.gz models/*.gz`
3. **(Optional) Generate results** — `python run_pipeline.py` to populate the dashboard tables and charts
4. **Launch the app** — `python app.py`
5. **Open** `http://127.0.0.1:5001` in your browser

### Using the Support Assistant (Chatbot)

The **Support Assistant** is the customer-facing page. It works like a chat:

1. Click **Support Assistant** in the top navigation.
2. Pick the **issue category** from the dropdown (Order Related, Returns, Refund Related, etc.).
3. Type your **query** — e.g. "My order hasn't arrived and it's been 10 days" — and press **Send**.
4. The bot replies instantly with:
   - A **greeting** confirming your issue category
   - An **intent-aware message** (e.g. it recognizes damage, delays, refund delays, cancellations)
   - **Numbered solution steps** you can follow
   - A **predicted sentiment** badge showing whether the interaction looks satisfied or dissatisfied

The assistant recognizes urgency: complaints about damaged items, missing deliveries, or unprocessed refunds are flagged as **high urgency** and get escalation messaging.

Example queries to try:
- "The product arrived broken and damaged" (Returns) → replacement/refund offer
- "My refund was never processed" (Refund Related) → priority escalation
- "How do I cancel my order?" (Cancellation) → cancellation steps

> Every chat is logged to `data/user_feedback_log.csv` for future retraining (see Continuous Learning below).

### Making a Prediction (Step by Step)

1. Click **Live Prediction** in the top navigation bar.
2. Fill in the interaction fields:
   - **Customer Message** — the text the customer wrote (this is the most important input)
   - **Category** — the type of issue (Returns, Refund Related, Order Related, etc.)
   - **Channel** — how the customer reached out (Inbound, Outcall, Email)
   - **Agent Shift** — when the agent handled it (Morning, Afternoon, Evening, Night, Split)
   - **Agent Tenure** — the agent's experience bucket (On Job Training, 0-30, 31-60, 61-90, >90 days)
   - **Response Time (minutes)** — how long before the customer got a response
   - **Hour of Day** — the hour (0-23) the issue was reported
3. Click **Predict Satisfaction**.
4. Read the result panel:
   - **Verdict** — Satisfied (Positive) or Dissatisfied (Negative)
   - **Probability bars** — likelihood of each outcome
   - **Confidence** — how sure the model is
   - **Processed text** — the cleaned message the model actually analyzed

### How to Interpret the Output

| Signal | Meaning |
|--------|---------|
| Verdict = Satisfied | The customer is likely to give a CSAT score of 4-5 |
| Verdict = Dissatisfied | The customer is likely to give a CSAT score of 1-3 |
| Confidence > 70% | Strong prediction — the message/features clearly point one way |
| Confidence 50-60% | Borderline case — treat with caution, consider human review |
| Processed text is empty | The message had no usable words after cleaning; prediction relies on structured features only |

### Sample Inputs (Where the Model Works Well)

These examples are known to produce reliable predictions. Try them in the Live Prediction form:

**Example 1 — Clear negative (works well)**
```
Message: "My order arrived late and the item was damaged. Very disappointed with the service."
Category: Returns | Channel: Inbound | Shift: Morning | Tenure: 0-30
Response Time: 120 | Hour: 14
Expected: Dissatisfied (Negative), ~68% confidence
```

**Example 2 — Clear positive (works well)**
```
Message: "Excellent service! The agent was very helpful and resolved my issue quickly. Thank you so much!"
Category: Order Related | Channel: Inbound | Shift: Morning | Tenure: >90
Response Time: 5 | Hour: 10
Expected: Satisfied (Positive)
```

**Example 3 — Strong negative with slow response (works well)**
```
Message: "Terrible experience, my refund was never processed and no one responded."
Category: Refund Related | Channel: Email | Shift: Night | Tenure: 0-30
Response Time: 300 | Hour: 23
Expected: Dissatisfied (Negative), ~79% confidence
```

### Ideal Approach for Best Results

- **Write a real, descriptive message.** The customer message drives most of the prediction. One-word or empty messages give weak, low-confidence results.
- **Use realistic response times.** Very fast responses (under 15 min) push toward satisfied; long delays (over 2 hours) push toward dissatisfied.
- **Match the category to the message.** A refund complaint should use "Refund Related", not "Feedback".
- **Treat borderline results (confidence 50-60%) as "needs human review"** rather than a firm decision.
- **For production accuracy**, run `python run_pipeline.py` to train and persist the optimized models before serving predictions.

### Continuous Learning (Feedback Logging)

Every interaction made through the web app — both **Support Assistant** chats (`/api/chat`) and **Live Prediction** submissions (`/api/predict`) — is automatically appended to:

```
data/user_feedback_log.csv
```

Each row records the submitted message, all interaction fields, and the model's prediction with its confidence. Over time this builds a growing log of real-world inputs that can be:

- Reviewed to spot cases where the model struggles
- Labeled with the true CSAT outcome and **merged into the training dataset**
- Used to **retrain and improve** the models in a future pipeline run

To incorporate the accumulated feedback into training, append the reviewed/labeled rows to the main dataset (`data/cumulative_ai_customer_communication_dataset.csv`) and re-run `python run_pipeline.py`.

> The feedback log is git-ignored by default (it is runtime user data). Remove `data/*.csv` from `.gitignore` if you want to version it.

### Where to Find Results

After running, outputs are saved to:
```
models/                          # Trained model files (.pkl)
├── optimized_xgb_grid.pkl       # Best XGBoost from GridSearch
├── optimized_xgb_bayesian.pkl   # Best XGBoost from Optuna
├── optimized_lr.pkl             # Best Logistic Regression
├── optimized_rf.pkl             # Best Random Forest
├── sbert_train_embeddings.npy   # Sentence-BERT embeddings
└── optimization_results.json    # All tuning results

results/                         # Plots and reports
├── pipeline_results.json        # Full structured results
├── evaluation_results.csv       # Model metrics table
├── confusion_matrices.png       # Confusion matrices for all models
├── roc_pr_curves.png            # ROC and Precision-Recall curves
├── cv_comparison.png            # Cross-validation boxplots
├── feature_importance.png       # XAI feature rankings
├── shap_summary.png             # SHAP global explanations
├── learning_curves.png          # Train vs validation curves
└── error_analysis.png           # Where models fail
```

## Key Results

### Dataset Overview
- **Total samples**: 85,907 customer interactions
- **Target distribution**: Positive (CSAT 4-5) = 82.5%, Negative (CSAT 1-3) = 17.5%
- **Train/Test split**: 80/20 stratified

### Model Performance Comparison

| Model | Feature Basis | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|--------------|----------|-----------|--------|----------|---------|
| SVM (TF-IDF) | TF-IDF text | 85.10% | 85.51% | 98.65% | 91.61% | 0.6911 |
| Naive Bayes (TF-IDF) | TF-IDF text | 84.97% | 85.50% | 98.48% | 91.53% | 0.6970 |
| Gradient Boosting (Struct) | Structured features | 83.34% | 84.42% | 97.85% | 90.64% | 0.7353 |
| LR (TF-IDF) | TF-IDF text | 83.59% | 87.14% | 93.97% | 90.43% | 0.6992 |
| Random Forest (TF-IDF) | TF-IDF text | 82.66% | 86.92% | 92.95% | 89.84% | 0.6932 |
| LR (Combined) | TF-IDF + Structured | 74.61% | 88.85% | 79.14% | 83.71% | 0.7273 |
| LR (Structured) | Structured features | 69.04% | 87.05% | 73.38% | 79.63% | 0.6555 |
| XGBoost (Structured) | Structured features | 69.71% | 89.98% | 71.19% | 79.49% | 0.7311 |

### What Each Model Prioritizes

| Model | Decision Basis | Why It Works |
|-------|---------------|--------------|
| SVM (TF-IDF) | Finds optimal hyperplane in TF-IDF text space to separate sentiment | Best F1 - captures textual cues of satisfaction/dissatisfaction in customer messages |
| Naive Bayes (TF-IDF) | Probability of word occurrence given satisfaction class | Fast, effective for text — leverages word frequency patterns |
| Gradient Boosting | Iteratively corrects errors using structured operational features | Best AUC (0.7353) — captures non-linear interactions in response time, category |
| LR (TF-IDF) | Linear combination of TF-IDF word weights | Interpretable text model — identifies key positive/negative word signals |
| Random Forest (TF-IDF) | Ensemble of decision trees on text features | Handles high-dimensional sparse text well with bagging |
| XGBoost (Structured) | Sequential boosting on engineered features | Highest precision (89.98%) — conservative positive predictions but best at AUC discrimination |

### Feature Importance (XGBoost - What Drives Predictions)

| Rank | Feature | Importance | Interpretation |
|------|---------|-----------|----------------|
| 1 | message_length | 0.2099 | Longer messages correlate with complaint complexity |
| 2 | response_time_minutes | 0.1518 | Faster response strongly predicts satisfaction |
| 3 | has_message | 0.1494 | Whether customer left a message indicates engagement |
| 4 | category_encoded | 0.1030 | Issue category (Returns, Refunds, etc.) drives CSAT |
| 5 | subcategory_encoded | 0.0838 | Specific issue type refines prediction |
| 6 | channel_encoded | 0.0630 | Communication channel affects satisfaction |
| 7 | shift_encoded | 0.0445 | Agent shift timing matters for resolution quality |
| 8 | tenure_encoded | 0.0443 | Agent experience level correlates with outcomes |
| 9 | issue_hour | 0.0407 | Time of day affects service quality |
| 10 | issue_day_of_week | 0.0385 | Weekday patterns in satisfaction |
| 11 | cleaned_word_count | 0.0385 | Word count after NLP cleaning |
| 12 | word_count | 0.0325 | Raw word count in message |

### Transformer Benchmarking Results (BERT / Sentence-BERT)

| Model | Embedding | Accuracy | F1-Score | AUC-ROC |
|-------|-----------|----------|----------|---------|
| XGBoost + Sentence-BERT | all-MiniLM-L6-v2 (384-dim) | **84.39%** | **90.96%** | 0.7025 |
| LR + Sentence-BERT | all-MiniLM-L6-v2 (384-dim) | 82.93% | 89.95% | **0.7115** |
| LR + TF-IDF (Baseline) | TF-IDF (5000-dim) | 83.59% | 90.43% | 0.6992 |

Sentence-BERT provides dense 384-dimensional embeddings that capture semantic meaning better than sparse TF-IDF vectors. The SBERT+XGBoost combination achieves competitive F1 while using 13x fewer dimensions than TF-IDF.

### Hyperparameter Optimization Results

| Method | Best CV F1 | Test F1 | Best Parameters | Time |
|--------|-----------|---------|-----------------|------|
| Bayesian (Optuna, 30 trials) | **84.15%** | **83.68%** | n_estimators=496, max_depth=11, lr=0.178 | 34.2s |
| RandomizedSearchCV (50 iter) | 83.57% | 83.25% | n_estimators=436, max_depth=9, lr=0.421 | 30.0s |
| GridSearchCV | 81.38% | 80.78% | n_estimators=300, max_depth=7, lr=0.2 | 11.3s |

Bayesian optimization (Optuna TPE sampler) outperforms both grid and random search by intelligently exploring the parameter space, achieving 2.8% higher F1 than GridSearch.

### Cross-Validation Stability (10-Fold)

| Model | Mean F1 | Std Dev | Stability |
|-------|---------|---------|-----------|
| Gradient Boosting | **90.51%** | +/-0.12% | Most stable |
| Random Forest | 89.88% | +/-0.25% | Stable |
| Logistic Regression | 83.36% | +/-1.57% | Moderate variance |
| XGBoost | 79.65% | +/-0.28% | Stable (lower F1 due to class weight) |

### Key Observations

1. **Text-based models dominate on F1-Score**: SVM and Naive Bayes with TF-IDF achieve the highest F1 (91.6%) because customer message text directly contains satisfaction/dissatisfaction signals.

2. **Structured models lead on AUC-ROC**: Gradient Boosting (0.7353) and XGBoost (0.7311) are better at discriminating between classes overall, even though their accuracy is lower — they're more conservative and don't over-predict the majority class.

3. **Transformer embeddings improve discrimination**: Sentence-BERT achieves AUC 0.7115, outperforming TF-IDF baseline (0.6992), indicating better semantic understanding of customer messages.

4. **Class imbalance effect**: With 82.5% positive class, text models that predict "positive" more aggressively get high recall (98%+) and thus high F1, but their AUC reveals limited true discrimination ability.

5. **Bayesian optimization finds better solutions**: Optuna's TPE sampler achieves 2.8% higher F1 than exhaustive GridSearch in similar time, validating the sequential optimization approach.

6. **response_time_minutes** is the strongest operational predictor — faster response directly improves satisfaction.

7. **message_length** is the top XGBoost feature — longer messages typically indicate more complex/negative issues.

### Output Files

All results from `run_pipeline.py` are saved to `results/` and `models/` directories (see "Where to Find Results" above).

## Technical Details

### Hyperparameter Optimization
- **GridSearchCV**: Exhaustive search over defined parameter grids
- **RandomizedSearchCV**: 50-100 iterations sampling from distributions
- **Bayesian Optimization (Optuna)**: TPE sampler, 30-50 trials with convergence tracking

### Statistical Validation
- **10-Fold Stratified Cross-Validation** for stability assessment
- **Friedman Test** for overall model ranking significance
- **Paired t-test** for pairwise model comparison
- **McNemar's Test** for classification disagreement

### Advanced Features
- Communication Risk Index (CRI) - novel composite metric
- Dynamic Multi-Modal Fusion - context-aware feature weighting
- Adaptive Feature Selection - per-segment optimization
- VADER Sentiment Analysis integration

## Requirements

- Python 3.9+
- See `requirements.txt` for full dependency list
- GPU optional (speeds up transformer inference)

## Author

MT24AAC019 - Major Project

## License

Academic use only.
