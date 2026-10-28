"""
Knowledge base for the support system.

Stores admin-approved question/answer pairs. When a customer asks a question:
  1. We look for a similar answered question in the KB and return that answer.
  2. If nothing matches, the query is escalated to the admin's pending queue.

When an admin answers a pending query, the Q&A is appended to the KB so the
same (or similar) question is auto-answered next time — with the real answer,
not a random canned response.

Similarity uses TF-IDF cosine over stored questions (falls back to token
overlap if scikit-learn is unavailable).
"""
import os
import json
import re
import threading
from datetime import datetime

from . import storage

# Runtime files live in storage.RUNTIME_DIR (data/ by default). All reads and
# writes go through storage, which backs up, refuses shrinking writes, and
# never treats a corrupt file as empty.
DATA_DIR = storage.RUNTIME_DIR
KB_FILE = storage.runtime_path("knowledge_base.json")
PENDING_FILE = storage.runtime_path("pending_queries.json")

# Minimum cosine similarity for a KB hit (0-1). Tuned so paraphrased questions
# ("Do you offer gift wrapping?" vs "Can I get gift wrapping?") still match,
# while unrelated questions escalate to the admin.
MATCH_THRESHOLD = 0.30

# Reentrant lock so nested lock acquisitions (e.g. a helper that also locks)
# never deadlock.
_lock = threading.RLock()

# A small hardcoded fallback used only if the dataset can't be read. On a normal
# install the knowledge base is built from the dataset (see _ensure_seed_kb).
_FALLBACK_KB = [
    {"question": "How do I track my order?",
     "answer": "You can track your order in real time under My Orders > Track.",
     "category": "Order Related"},
    {"question": "How do I return an item?",
     "answer": "Go to My Orders > Return Item within the return window (7-10 days).",
     "category": "Returns"},
    {"question": "When will I get my refund?",
     "answer": "Refunds reach your original payment method 5-7 business days after "
               "the returned item is received and inspected.",
     "category": "Refund Related"},
]


# ---------------------------------------------------------------------------
# Low-level JSON helpers
# ---------------------------------------------------------------------------
def _read_json(path, default):
    # Corrupt files are restored from backup, never read as empty.
    return storage.read_json(path, default)


def _write_json(path, data):
    # Backs up the previous version and refuses writes that drop records.
    storage.write_json(path, data)


def _normalize(text):
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------
def _token_overlap(a, b):
    """Jaccard token overlap fallback."""
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _best_match(query, entries):
    """
    Return (best_entry, score) for the query against KB entries.
    Uses TF-IDF cosine when possible, else token overlap.
    """
    if not entries:
        return None, 0.0

    q_norm = _normalize(query)
    questions = [_normalize(e["question"]) for e in entries]

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        corpus = questions + [q_norm]
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        matrix = vec.fit_transform(corpus)
        sims = cosine_similarity(matrix[-1], matrix[:-1])[0]
        best_idx = int(sims.argmax())
        return entries[best_idx], float(sims[best_idx])
    except Exception:
        scores = [_token_overlap(q_norm, qi) for qi in questions]
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        return entries[best_idx], scores[best_idx]


# ---------------------------------------------------------------------------
# Knowledge base API
# ---------------------------------------------------------------------------
def _ensure_seed_kb():
    """
    Create knowledge_base.json the first time it is needed, built from the
    dataset's satisfied interactions (csat_score >= 4), one verified entry per
    sub-category topic. Falls back to a few hardcoded entries if the dataset
    can't be read. Only runs when the file does not exist, so admin-added
    answers are never overwritten.
    """
    if os.path.exists(KB_FILE):
        return
    with _lock:
        if os.path.exists(KB_FILE):  # re-check inside lock
            return

        # Build verified entries from the dataset's satisfied interactions.
        built = []
        try:
            from . import kb_builder
            built = kb_builder.build_entries()
        except Exception:
            built = []

        if not built:
            # Dataset unavailable — use the minimal fallback.
            now = datetime.now().isoformat(timespec="seconds")
            built = [{**item, "source": "fallback", "created_at": now}
                     for item in _FALLBACK_KB]

        seeded = []
        for i, item in enumerate(built, 1):
            seeded.append({
                "id": i,
                "question": item["question"],
                "answer": item["answer"],
                "category": item["category"],
                "source": item.get("source", "dataset"),
                "support": item.get("support"),
                "created_at": item.get("created_at",
                                       datetime.now().isoformat(timespec="seconds")),
            })
        _write_json(KB_FILE, seeded)


def load_kb():
    _ensure_seed_kb()
    return _read_json(KB_FILE, [])


def lookup(query, category=None, threshold=MATCH_THRESHOLD):
    """
    Find the best KB answer for a query.

    Returns dict {matched: bool, answer, question, score, category} —
    matched=False means no confident match was found.
    """
    entries = load_kb()
    if category:
        scoped = [e for e in entries if e.get("category") == category]
        entries_to_search = scoped if scoped else entries
    else:
        entries_to_search = entries

    best, score = _best_match(query, entries_to_search)
    if best and score >= threshold:
        return {
            "matched": True,
            "answer": best["answer"],
            "question": best["question"],
            "score": round(score, 3),
            "category": best.get("category"),
        }
    return {"matched": False, "answer": None, "question": None,
            "score": round(score, 3), "category": category}


def add_entry(question, answer, category, source="admin"):
    """Append a new answered Q&A to the knowledge base (persisted to disk)."""
    with _lock:
        entries = load_kb()
        next_id = (max((e["id"] for e in entries), default=0)) + 1
        entries.append({
            "id": next_id,
            "question": question.strip(),
            "answer": answer.strip(),
            "category": category,
            "source": source,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        })
        _write_json(KB_FILE, entries)
    return entries[-1]


# ---------------------------------------------------------------------------
# Pending (escalated) queries API
# ---------------------------------------------------------------------------
def load_pending():
    return _read_json(PENDING_FILE, [])


def add_pending(query, category, customer=None, conversation=None):
    """
    Escalate an unanswered query to the admin queue.
    Avoids duplicates: if an identical open query by the same customer exists,
    reuse it. Stores the full chat transcript so it is retained after logout.
    """
    with _lock:
        pending = load_pending()
        q_norm = _normalize(query)
        for p in pending:
            if p["status"] == "open" and _normalize(p["query"]) == q_norm \
                    and p.get("category") == category \
                    and p.get("customer") == (customer or "anonymous"):
                return p  # already queued for this customer

        entry = {
            "id": (max([p["id"] for p in pending]) + 1) if pending else 1,
            "query": query.strip(),
            "category": category,
            "customer": customer or "anonymous",
            "status": "open",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "answer": None,
            "answered_at": None,
            # Full chat transcript — list of {role, text, at}
            "conversation": conversation or [
                {"role": "customer", "text": query.strip(),
                 "at": datetime.now().isoformat(timespec="seconds")}
            ],
            # Set True once the customer has seen the agent's answer, so we only
            # show the "agent replied" notification once.
            "customer_seen": False,
        }
        pending.append(entry)
        _write_json(PENDING_FILE, pending)
    return entry


def answer_pending(pending_id, answer, admin=None):
    """
    Admin answers a pending query. Marks it resolved and promotes the Q&A
    into the knowledge base so future matching queries auto-answer.
    """
    with _lock:
        pending = load_pending()
        target = None
        for p in pending:
            if p["id"] == int(pending_id):
                target = p
                break
        if target is None:
            return None

        target["status"] = "answered"
        target["answer"] = answer.strip()
        target["answered_at"] = datetime.now().isoformat(timespec="seconds")
        target["answered_by"] = admin or "admin"
        target["customer_seen"] = False  # customer hasn't seen this reply yet
        # Append the agent's answer to the chat transcript
        target.setdefault("conversation", []).append({
            "role": "agent",
            "text": answer.strip(),
            "at": target["answered_at"],
        })
        _write_json(PENDING_FILE, pending)

    # Promote to KB (outside lock section re-acquires inside add_entry)
    add_entry(target["query"], answer, target["category"], source="admin")
    return target


def pending_counts():
    pending = load_pending()
    return {
        "open": sum(1 for p in pending if p["status"] == "open"),
        "answered": sum(1 for p in pending if p["status"] == "answered"),
        "total": len(pending),
    }


# ---------------------------------------------------------------------------
# Customer-facing ticket API (chat history retained across logout)
# ---------------------------------------------------------------------------
def queries_for_customer(username):
    """Return all tickets raised by a customer, newest first."""
    pending = load_pending()
    mine = [p for p in pending if p.get("customer") == username]
    return sorted(mine, key=lambda p: p["id"], reverse=True)


def get_pending(ticket_id):
    """Return a single ticket by id, or None."""
    for p in load_pending():
        if p["id"] == int(ticket_id):
            return p
    return None


def append_message(ticket_id, role, text):
    """Append a message to a ticket's conversation transcript."""
    with _lock:
        pending = load_pending()
        for p in pending:
            if p["id"] == int(ticket_id):
                p.setdefault("conversation", []).append({
                    "role": role,
                    "text": text,
                    "at": datetime.now().isoformat(timespec="seconds"),
                })
                _write_json(PENDING_FILE, pending)
                return p
    return None


def mark_customer_seen(ticket_id):
    """Mark that the customer has viewed the agent's answer for this ticket."""
    with _lock:
        pending = load_pending()
        for p in pending:
            if p["id"] == int(ticket_id):
                p["customer_seen"] = True
                _write_json(PENDING_FILE, pending)
                return p
    return None


def customer_notifications(username):
    """
    Return tickets for this customer that were answered but not yet seen —
    used to show the 'your query was answered' banner.
    """
    answered_unseen = [
        p for p in load_pending()
        if p.get("customer") == username
        and p["status"] == "answered"
        and not p.get("customer_seen", False)
    ]
    return sorted(answered_unseen, key=lambda p: p["id"], reverse=True)
