#!/usr/bin/env python3
"""
Flask web application for the Ecommerce AI Communication Framework.

Provides a dashboard to:
  - View model comparison, optimization, transformer benchmarking, and XAI results
  - Run live customer-satisfaction predictions
  - Browse generated visualizations

Run with:
    python app.py
Then open http://127.0.0.1:5000
"""
import os
import json
from flask import Flask, render_template, request, jsonify, send_from_directory

from src.predictor import (
    get_predictor, log_feedback, CATEGORY_OPTIONS, CHANNEL_OPTIONS,
    SHIFT_OPTIONS, TENURE_OPTIONS,
)
from src.support_bot import generate_response, SUPPORT_CATEGORIES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RESULTS_JSON = os.path.join(RESULTS_DIR, "pipeline_results.json")

app = Flask(__name__)


def load_results():
    """Load pipeline results JSON, return None if not yet generated."""
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON) as f:
            return json.load(f)
    return None


def list_result_images():
    """Return available PNG visualizations in results/."""
    if not os.path.isdir(RESULTS_DIR):
        return []
    imgs = [f for f in os.listdir(RESULTS_DIR) if f.lower().endswith(".png")]
    labels = {
        "confusion_matrices.png": "Confusion Matrices",
        "roc_pr_curves.png": "ROC & Precision-Recall Curves",
        "cv_comparison.png": "Cross-Validation Comparison",
        "feature_importance.png": "Feature Importance (XAI)",
        "shap_summary.png": "SHAP Summary",
        "learning_curves.png": "Learning Curves",
        "error_analysis.png": "Error Analysis",
        "optimization_results.png": "Optimization Convergence",
        "transformer_comparison.png": "Transformer Benchmarking",
        "comparative_analysis.png": "Comparative Analysis",
    }
    return sorted(
        [{"file": f, "label": labels.get(f, f.replace("_", " ").replace(".png", "").title())}
         for f in imgs],
        key=lambda x: x["label"],
    )


@app.route("/")
def index():
    results = load_results()
    return render_template("index.html", results=results, active="dashboard")


@app.route("/predict", methods=["GET"])
def predict_page():
    return render_template(
        "predict.html",
        active="predict",
        categories=CATEGORY_OPTIONS,
        channels=CHANNEL_OPTIONS,
        shifts=SHIFT_OPTIONS,
        tenures=TENURE_OPTIONS,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON prediction endpoint. Also logs each interaction for future retraining."""
    data = request.get_json(force=True, silent=True) or request.form
    inputs = {
        "message": data.get("message", ""),
        "category": data.get("category", CATEGORY_OPTIONS[0]),
        "channel": data.get("channel", CHANNEL_OPTIONS[0]),
        "shift": data.get("shift", SHIFT_OPTIONS[0]),
        "tenure": data.get("tenure", TENURE_OPTIONS[0]),
        "response_time_minutes": float(data.get("response_time_minutes", 30) or 30),
        "issue_hour": int(data.get("issue_hour", 12) or 12),
        "issue_day_of_week": int(data.get("issue_day_of_week", 2) or 2),
    }
    try:
        predictor = get_predictor()
        result = predictor.predict(**inputs)

        # Persist the interaction + prediction so the dataset grows over time.
        try:
            log_feedback(inputs, result)
        except Exception as log_err:
            app.logger.warning("Feedback logging failed: %s", log_err)

        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/support", methods=["GET"])
def support_page():
    """Customer-facing support bot page."""
    return render_template(
        "support.html", active="support", categories=SUPPORT_CATEGORIES
    )


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Support bot endpoint: takes a customer query + category, returns a solution
    plus a satisfaction prediction. Logs the interaction for future retraining.
    """
    data = request.get_json(force=True, silent=True) or request.form
    query = data.get("query", "") or data.get("message", "")
    category = data.get("category", SUPPORT_CATEGORIES[0])

    try:
        # 1. Generate the support solution
        bot = generate_response(query, category)

        # 2. Predict likely customer satisfaction from the query
        prediction = None
        try:
            predictor = get_predictor()
            prediction = predictor.predict(
                message=query,
                category=category,
                channel=CHANNEL_OPTIONS[0],
                shift=SHIFT_OPTIONS[0],
                tenure=TENURE_OPTIONS[0],
                response_time_minutes=30,
            )
            # Log the interaction so the dataset grows over time
            log_feedback(
                {
                    "message": query, "category": category,
                    "channel": CHANNEL_OPTIONS[0], "shift": SHIFT_OPTIONS[0],
                    "tenure": TENURE_OPTIONS[0], "response_time_minutes": 30,
                    "issue_hour": 12, "issue_day_of_week": 2,
                },
                prediction,
            )
        except Exception as pred_err:
            app.logger.warning("Prediction/logging failed in chat: %s", pred_err)

        return jsonify({"success": True, "bot": bot, "prediction": prediction})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/visualizations")
def visualizations():
    return render_template(
        "visualizations.html", active="viz", images=list_result_images()
    )


@app.route("/results-image/<path:filename>")
def results_image(filename):
    return send_from_directory(RESULTS_DIR, filename)


@app.route("/api/results")
def api_results():
    """Raw results JSON endpoint."""
    results = load_results()
    if results is None:
        return jsonify({"error": "No results found. Run python run_pipeline.py first."}), 404
    return jsonify(results)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # Port 5000 conflicts with macOS AirPlay Receiver, so default to 5001.
    # Override with: PORT=8080 python app.py
    port = int(os.environ.get("PORT", 5001))
    base = f"http://127.0.0.1:{port}"
    print("=" * 60)
    print("  Ecommerce AI Communication Framework - Web UI")
    print("=" * 60)
    print(f"  Dashboard:       {base}")
    print(f"  Support Bot:     {base}/support")
    print(f"  Live Predict:    {base}/predict")
    print(f"  Visualizations:  {base}/visualizations")
    print("=" * 60)
    # debug=False avoids the reloader spawning an orphan child process that
    # can keep squatting on the port after the server is stopped.
    app.run(host="0.0.0.0", port=port, debug=False)
