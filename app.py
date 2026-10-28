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
from src.support_bot import (
    generate_response, get_clarifiers, recommendation_for, SUPPORT_CATEGORIES,
)
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
    """
    Role-based landing page. The ML dashboard is admin-only:
      - not logged in -> login page
      - customer      -> support assistant
      - admin         -> dashboard
    """
    user = auth.current_user()
    if user is None:
        return redirect(url_for("login_page"))
    if user["role"] != "admin":
        return redirect(url_for("support_page"))
    return render_template("index.html", results=load_results(), active="dashboard")


@app.route("/predict", methods=["GET"])
@auth.login_required(role="admin")
def predict_page():
    return render_template(
        "predict.html", active="predict",
        categories=CATEGORY_OPTIONS, channels=CHANNEL_OPTIONS,
        shifts=SHIFT_OPTIONS, tenures=TENURE_OPTIONS,
    )


@app.route("/api/predict", methods=["POST"])
@auth.login_required(role="admin")
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
@auth.login_required(role="admin")
def visualizations():
    return render_template("visualizations.html", active="viz", images=list_result_images())


@app.route("/results-image/<path:filename>")
@auth.login_required(role="admin")
def results_image(filename):
    return send_from_directory(RESULTS_DIR, filename)


@app.route("/api/results")
@auth.login_required(role="admin")
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
    user = auth.current_user() or {}
    customer = user.get("username", "anonymous")
    my_queries = kb.queries_for_customer(customer)
    return render_template(
        "support.html", active="support",
        categories=SUPPORT_CATEGORIES,
        my_queries=my_queries,
    )


@app.route("/api/chat", methods=["POST"])
@auth.login_required(role="customer")
def api_chat():
    """
    Staged, guided support flow (never a random reply, agent is the last resort):

      Stage "ask" (default): customer sends a question.
        - If the knowledge base has a confident match -> return the verified
          answer as a recommendation, then ask if it resolved their issue.
        - If no match -> start the clarifying-questions flow (stage "clarify").

      Stage "clarify": the bot asks 1-2 clarifying questions one at a time.
        After re-checking the KB with the added context and still finding
        nothing, it moves to stage "exhausted".

      Stage "exhausted": the bot has tried everything it can; it apologises and
        offers the "Chat with an Agent" button. Escalation happens only here,
        and only when the customer clicks the button (handled by /api/escalate).

    The client passes the flow state back each turn:
        {query, category, stage, clarify_step, original_query, collected[]}
    """
    data = request.get_json(force=True, silent=True) or request.form
    query = (data.get("query", "") or data.get("message", "")).strip()
    category = data.get("category", SUPPORT_CATEGORIES[0])
    stage = data.get("stage", "ask")
    clarify_step = int(data.get("clarify_step", 0) or 0)
    original_query = (data.get("original_query", "") or query).strip()
    collected = data.get("collected") or []
    if isinstance(collected, str):
        collected = [collected]

    user = auth.current_user() or {}
    customer = user.get("username", "anonymous")

    if not query:
        return jsonify({"success": True, "bot": {
            "source": "system", "stage": "ask",
            "reply": "Please describe your issue so I can help.",
        }, "prediction": None})

    def _predict_and_log(text):
        try:
            predictor = get_predictor()
            pred = predictor.predict(
                message=text, category=category,
                channel=CHANNEL_OPTIONS[0], shift=SHIFT_OPTIONS[0],
                tenure=TENURE_OPTIONS[0], response_time_minutes=30,
            )
            log_feedback({
                "message": text, "category": category,
                "channel": CHANNEL_OPTIONS[0], "shift": SHIFT_OPTIONS[0],
                "tenure": TENURE_OPTIONS[0], "response_time_minutes": 30,
                "issue_hour": 12, "issue_day_of_week": 2,
            }, pred)
            return pred
        except Exception as pred_err:
            app.logger.warning("Prediction/logging failed in chat: %s", pred_err)
            return None

    try:
        clarifiers = get_clarifiers(category)
        prediction = None

        # ---- Stage: customer responding to "Did this resolve your issue?" ----
        if stage == "resolved":
            ans = query.strip().lower()
            positive = any(w in ans for w in [
                "yes", "yeah", "yep", "resolved", "solved", "thanks", "thank",
                "great", "perfect", "got it", "helps", "helped", "ok", "okay",
            ])
            if positive:
                bot = {
                    "source": "system", "stage": "ask",
                    "reply": "Glad I could help! Is there anything else I can assist with?",
                    "offer_agent": False,
                }
                return jsonify({"success": True, "bot": bot, "prediction": None})
            # Not resolved -> go straight to clarify (bot keeps trying before agent)
            stage = "ask"
            original_query = original_query or query
            query = original_query

        # ---- Stage: initial ask (or still in clarify, re-check KB) ----
        if stage in ("ask", "clarify"):
            # Combine the original question with any clarifying answers so far
            # to give the KB lookup more to work with.
            combined = " ".join([original_query] + collected + [query]).strip()
            match = kb.lookup(combined, category=category)

            if match["matched"]:
                prediction = _predict_and_log(combined)
                bot = {
                    "source": "knowledge_base",
                    "stage": "resolved",
                    "reply": recommendation_for(category, match["answer"]),
                    "matched_question": match["question"],
                    "score": match["score"],
                    "follow_up": "Did this resolve your issue?",
                    "offer_agent": False,
                }
                return jsonify({"success": True, "bot": bot, "prediction": None})

            # No KB match. Walk through clarifying questions one at a time.
            if stage == "ask":
                # First miss: open the clarify flow with recommendation + Q1
                prediction = _predict_and_log(original_query)
                rec = recommendation_for(category, None)
                bot = {
                    "source": "clarify",
                    "stage": "clarify",
                    "reply": (
                        f"{rec}\n\nTo help pinpoint this, {clarifiers[0]}"
                    ),
                    "clarify_step": 1,
                    "original_query": original_query,
                    "collected": [],
                    "offer_agent": False,
                }
                return jsonify({"success": True, "bot": bot, "prediction": None})

            # stage == "clarify": record this answer and ask the next question
            collected = collected + [query]
            if clarify_step < len(clarifiers):
                bot = {
                    "source": "clarify",
                    "stage": "clarify",
                    "reply": f"Thanks. {clarifiers[clarify_step]}",
                    "clarify_step": clarify_step + 1,
                    "original_query": original_query,
                    "collected": collected,
                    "offer_agent": False,
                }
                return jsonify({"success": True, "bot": bot, "prediction": None})

            # Clarifiers exhausted and still no verified answer -> last resort
            prediction = _predict_and_log(" ".join([original_query] + collected))
            bot = {
                "source": "exhausted",
                "stage": "exhausted",
                "reply": (
                    "Thanks for the details. I've shared everything I can from our "
                    "verified help, but I don't have a confirmed answer for your "
                    "specific case. I'm sorry I couldn't fully resolve this — let me "
                    "connect you with a support agent who can."
                ),
                "offer_agent": True,
                "query": " ".join([original_query] + collected),
                "category": category,
            }
            return jsonify({"success": True, "bot": bot, "prediction": None})

        # ---- Fallback ----
        return jsonify({"success": True, "bot": {
            "source": "system", "stage": "ask",
            "reply": "How else can I help you?",
        }, "prediction": None})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/my-queries")
@auth.login_required(role="customer")
def my_queries_page():
    """Customer page listing their answered (addressed) and waiting queries."""
    user = auth.current_user() or {}
    customer = user.get("username", "anonymous")
    queries = kb.queries_for_customer(customer)
    answered = [q for q in queries if q["status"] == "answered"]
    waiting = [q for q in queries if q["status"] != "answered"]
    # Viewing this page counts as seeing the agent replies
    for q in answered:
        if not q.get("customer_seen", False):
            kb.mark_customer_seen(q["id"])
    return render_template(
        "my_queries.html", active="my_queries",
        answered=answered, waiting=waiting,
    )


@app.route("/api/escalate", methods=["POST"])
@auth.login_required(role="customer")
def api_escalate():
    """
    Customer explicitly chose to chat with an agent. Create a support ticket
    so the admin is notified. Called only when the user clicks the
    'Chat with an Agent' button — never automatically.
    """
    data = request.get_json(force=True, silent=True) or request.form
    query = (data.get("query", "") or "").strip()
    category = data.get("category", SUPPORT_CATEGORIES[0])
    user = auth.current_user() or {}
    customer = user.get("username", "anonymous")

    if not query:
        return jsonify({"success": False, "error": "No query to escalate"}), 400

    pending = kb.add_pending(query, category, customer=customer)
    return jsonify({
        "success": True,
        "ticket_id": pending["id"],
        "status": pending["status"],
        "reply": (
            "Thanks — I've connected you with a support agent and shared your "
            "question. An agent will respond here shortly. Your conversation is "
            "saved, so you can log out and come back to it anytime."
        ),
    })


@app.route("/api/ticket/<int:ticket_id>")
@auth.login_required(role="customer")
def api_ticket_status(ticket_id):
    """Let a customer poll whether their escalated ticket has been answered."""
    user = auth.current_user() or {}
    customer = user.get("username", "anonymous")
    p = kb.get_pending(ticket_id)
    if p and p.get("customer") == customer:
        return jsonify({
            "success": True,
            "status": p["status"],
            "answer": p.get("answer"),
            "query": p["query"],
        })
    return jsonify({"success": False, "error": "Ticket not found"}), 404


@app.route("/api/my-queries")
@auth.login_required(role="customer")
def api_my_queries():
    """List all of this customer's past queries (retained across logout)."""
    user = auth.current_user() or {}
    customer = user.get("username", "anonymous")
    queries = kb.queries_for_customer(customer)
    return jsonify({
        "success": True,
        "queries": [
            {
                "id": q["id"], "query": q["query"], "category": q["category"],
                "status": q["status"], "created_at": q["created_at"],
                "answered": q["status"] == "answered",
                "seen": q.get("customer_seen", False),
            }
            for q in queries
        ],
    })


@app.route("/api/my-queries/<int:ticket_id>")
@auth.login_required(role="customer")
def api_my_query_detail(ticket_id):
    """Return the full conversation for one of the customer's tickets."""
    user = auth.current_user() or {}
    customer = user.get("username", "anonymous")
    p = kb.get_pending(ticket_id)
    if not p or p.get("customer") != customer:
        return jsonify({"success": False, "error": "Query not found"}), 404
    # Opening the ticket marks the agent's answer as seen
    if p["status"] == "answered" and not p.get("customer_seen", False):
        kb.mark_customer_seen(ticket_id)
    return jsonify({
        "success": True,
        "ticket": {
            "id": p["id"], "query": p["query"], "category": p["category"],
            "status": p["status"], "answer": p.get("answer"),
            "conversation": p.get("conversation", []),
            "created_at": p["created_at"], "answered_at": p.get("answered_at"),
        },
    })


@app.route("/api/customer-notifications")
@auth.login_required(role="customer")
def api_customer_notifications():
    """
    Return the customer's tickets that an agent answered but the customer
    hasn't opened yet — powers the 'your query was answered' banner.
    """
    user = auth.current_user() or {}
    customer = user.get("username", "anonymous")
    unseen = kb.customer_notifications(customer)
    return jsonify({
        "success": True,
        "count": len(unseen),
        "answered": [
            {"id": q["id"], "query": q["query"], "answer": q.get("answer"),
             "category": q["category"]}
            for q in unseen
        ],
    })


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


@app.route("/api/admin/notifications")
@auth.login_required(role="admin")
def api_admin_notifications():
    """
    Poll endpoint for the admin page. Returns the current open queries so the
    UI can show a popup when a new one arrives.
    """
    pending = kb.load_pending()
    open_q = [p for p in pending if p["status"] == "open"]
    return jsonify({
        "success": True,
        "open_count": len(open_q),
        "open_queries": [
            {"id": p["id"], "query": p["query"], "category": p["category"],
             "customer": p.get("customer", "anonymous"), "created_at": p["created_at"]}
            for p in open_q
        ],
    })


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
