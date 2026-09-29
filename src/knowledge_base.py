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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

KB_FILE = os.path.join(DATA_DIR, "knowledge_base.json")
PENDING_FILE = os.path.join(DATA_DIR, "pending_queries.json")

# Minimum cosine similarity for a KB hit (0-1). Tuned so paraphrased questions
# ("Do you offer gift wrapping?" vs "Can I get gift wrapping?") still match,
# while unrelated questions escalate to the admin.
MATCH_THRESHOLD = 0.30

_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Low-level JSON helpers
# ---------------------------------------------------------------------------
def _read_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


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
def load_kb():
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
    """Append a new answered Q&A to the knowledge base."""
    with _lock:
        entries = load_kb()
        entries.append({
            "id": len(entries) + 1,
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


def add_pending(query, category, customer=None):
    """
    Escalate an unanswered query to the admin queue.
    Avoids duplicates: if an identical open query exists, reuse it.
    """
    with _lock:
        pending = load_pending()
        q_norm = _normalize(query)
        for p in pending:
            if p["status"] == "open" and _normalize(p["query"]) == q_norm \
                    and p.get("category") == category:
                return p  # already queued

        entry = {
            "id": (max([p["id"] for p in pending]) + 1) if pending else 1,
            "query": query.strip(),
            "category": category,
            "customer": customer or "anonymous",
            "status": "open",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "answer": None,
            "answered_at": None,
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
