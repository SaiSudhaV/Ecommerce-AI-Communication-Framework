#!/usr/bin/env python3
"""
Flask web application for the Ecommerce AI Communication Framework.

Features:
  - Customer/Admin login (session-based)
  - Knowledge-base-driven support chat:
      * If a similar answered question exists, reply with the real stored answer.
      * Otherwise escalate the query to the admin queue and tell the customer
        a specialist will follow up.
  - Admin dashboard to answer escalated queries. Answering promotes the Q&A
    into the knowledge base so the same question auto-answers next time.
  - ML dashboard, live prediction, and visualizations (unchanged).

Run with:  python app.py   (defaults to http://127.0.0.1:5001)
"""
import os
import json
from flask import (
    Flask, render_template, request, jsonify, send_from_directory,
    redirect, url_for, session, flash,
)

from src.predictor import (
    get_predictor, log_feedback, CATEGORY_OPTIONS, CHANNEL_OPTIONS,
    SHIFT_OPTIONS, TENURE_OPTIONS,
)
from src.support_bot import generate_response, SUPPORT_CATEGORIES
from src import knowledge_base as kb
from src import auth

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RESULTS_JSON = os.path.join(RESULTS_DIR, "pipeline_results.json")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "ecommerce-ai-support-dev-key")


# ---------------------------------------------------------------------------
# Template context: expose current user to all templates
# ---------------------------------------------------------------------------
@app.context_processor
def inject_user():
    return {"current_user": auth.current_user()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_results():
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON) as f:
            return json.load(f)
    return None


def list_result_images():
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


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login_page():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        user = auth.verify_user(username, password)
        if user:
            auth.login_session(user)
            nxt = request.args.get("next") or request.form.get("next")
            if user["role"] == "admin":
                return redirect(nxt or url_for("admin_page"))
            return redirect(nxt or url_for("support_page"))
        flash("Invalid username or password.", "error")
    return render_template("login.html", active="login")


@app.route("/register", methods=["POST"])
def register():
    username = request.form.get("username", "")
    password = request.form.get("password", "")
    ok, msg = auth.register_user(username, password, role="customer")
    flash(msg, "success" if ok else "error")
    if ok:
        user = auth.verify_user(username, password)
        if user:
            auth.login_session(user)
            return redirect(url_for("support_page"))
    return redirect(url_for("login_page"))


@app.route("/logout")
def logout():
    auth.logout_session()
    return redirect(url_for("login_page"))


# ---------------------------------------------------------------------------
# Public / ML pages
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", results=load_results(), active="dashboard")


@app.route("/predict", methods=["GET"])
def predict_page():
    return render_template(
        "predict.html", active="predict",
        categories=CATEGORY_OPTIONS, channels=CHANNEL_OPTIONS,
        shifts=SHIFT_OPTIONS, tenures=TENURE_OPTIONS,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
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
        try:
            log_feedback(inputs, result)
        except Exception as log_err:
            app.logger.warning("Feedback logging failed: %s", log_err)
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/visualizations")
def visualizations():
    return render_template("visualizations.html", active="viz", images=list_result_images())


@app.route("/results-image/<path:filename>")
def results_image(filename):
    return send_from_directory(RESULTS_DIR, filename)


@app.route("/api/results")
def api_results():
    results = load_results()
    if results is None:
        return jsonify({"error": "No results found. Run python run_pipeline.py first."}), 404
    return jsonify(results)


# ---------------------------------------------------------------------------
# Customer support (KB-driven)
# ---------------------------------------------------------------------------
@app.route("/support", methods=["GET"])
@auth.login_required(role="customer")
def support_page():
    return render_template("support.html", active="support", categories=SUPPORT_CATEGORIES)


@app.route("/api/chat", methods=["POST"])
@auth.login_required(role="customer")
def api_chat():
    """
    KB-first support endpoint.
      1. Look for a confident match in the knowledge base -> return real answer.
      2. If no match, escalate to admin queue and return an escalation notice.
    Also attaches a satisfaction prediction and logs the interaction.
    """
    data = request.get_json(force=True, silent=True) or request.form
    query = (data.get("query", "") or data.get("message", "")).strip()
    category = data.get("category", SUPPORT_CATEGORIES[0])
    user = auth.current_user() or {}
    customer = user.get("username", "anonymous")

    if not query:
        return jsonify({"success": True, "bot": {
            "source": "system",
            "reply": "Please describe your issue so I can help.",
        }, "prediction": None})

    try:
        # 1) Knowledge base lookup
        match = kb.lookup(query, category=category)

        if match["matched"]:
            bot = {
                "source": "knowledge_base",
                "reply": match["answer"],
                "matched_question": match["question"],
                "score": match["score"],
                "escalated": False,
            }
        else:
            # 2) Escalate to admin
            pending = kb.add_pending(query, category, customer=customer)
            bot = {
                "source": "escalated",
                "reply": (
                    "Thanks for your question. I don't have a verified answer for "
                    "this yet, so I've forwarded it to our support specialist. "
                    "You'll get an accurate response once they reply — please check "
                    "back shortly."
                ),
                "escalated": True,
                "ticket_id": pending["id"],
                "status": pending["status"],
            }

        # Satisfaction prediction + logging (best effort)
        prediction = None
        try:
            predictor = get_predictor()
            prediction = predictor.predict(
                message=query, category=category,
                channel=CHANNEL_OPTIONS[0], shift=SHIFT_OPTIONS[0],
                tenure=TENURE_OPTIONS[0], response_time_minutes=30,
            )
            log_feedback({
                "message": query, "category": category,
                "channel": CHANNEL_OPTIONS[0], "shift": SHIFT_OPTIONS[0],
                "tenure": TENURE_OPTIONS[0], "response_time_minutes": 30,
                "issue_hour": 12, "issue_day_of_week": 2,
            }, prediction)
        except Exception as pred_err:
            app.logger.warning("Prediction/logging failed in chat: %s", pred_err)

        return jsonify({"success": True, "bot": bot, "prediction": prediction})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/ticket/<int:ticket_id>")
@auth.login_required(role="customer")
def api_ticket_status(ticket_id):
    """Let a customer poll whether their escalated ticket has been answered."""
    for p in kb.load_pending():
        if p["id"] == ticket_id:
            return jsonify({
                "success": True,
                "status": p["status"],
                "answer": p.get("answer"),
                "query": p["query"],
            })
    return jsonify({"success": False, "error": "Ticket not found"}), 404


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------
@app.route("/admin")
@auth.login_required(role="admin")
def admin_page():
    pending = kb.load_pending()
    open_q = [p for p in pending if p["status"] == "open"]
    answered_q = [p for p in pending if p["status"] == "answered"]
    return render_template(
        "admin.html", active="admin",
        open_queries=open_q, answered_queries=answered_q,
        counts=kb.pending_counts(), kb_size=len(kb.load_kb()),
    )


@app.route("/api/admin/answer", methods=["POST"])
@auth.login_required(role="admin")
def api_admin_answer():
    """Admin answers a pending query; it is promoted into the knowledge base."""
    data = request.get_json(force=True, silent=True) or request.form
    ticket_id = data.get("ticket_id")
    answer = (data.get("answer", "") or "").strip()
    if not ticket_id or not answer:
        return jsonify({"success": False, "error": "ticket_id and answer are required"}), 400
    user = auth.current_user() or {}
    result = kb.answer_pending(ticket_id, answer, admin=user.get("username", "admin"))
    if result is None:
        return jsonify({"success": False, "error": "Ticket not found"}), 404
    return jsonify({"success": True, "ticket": result, "kb_size": len(kb.load_kb())})


@app.route("/api/admin/kb", methods=["GET"])
@auth.login_required(role="admin")
def api_admin_kb():
    """Return the current knowledge base (admin only)."""
    return jsonify({"success": True, "entries": kb.load_kb()})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    base = f"http://127.0.0.1:{port}"
    print("=" * 60)
    print("  Ecommerce AI Communication Framework - Web UI")
    print("=" * 60)
    print(f"  Login:           {base}/login")
    print(f"  Dashboard:       {base}")
    print(f"  Support (cust.): {base}/support")
    print(f"  Admin queue:     {base}/admin")
    print(f"  Live Predict:    {base}/predict")
    print(f"  Visualizations:  {base}/visualizations")
    print("-" * 60)
    print("  Seed logins:  admin/admin123   customer/customer123")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=False)
