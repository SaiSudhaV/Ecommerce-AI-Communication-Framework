"""
Simple session-based authentication for the support system.

Two roles:
  - customer: can use the support chat
  - admin:    can answer escalated queries

Users are stored in data/users.json. Passwords are salted-hashed (SHA-256).
Default seed accounts are created on first use so the app is usable out of the box:
    admin    / admin123    (role: admin)
    customer / customer123 (role: customer)

This is a lightweight demo auth layer, not production-grade identity management.
"""
import os
import json
import hashlib
import secrets
import threading
from functools import wraps

from flask import session, redirect, url_for, request, jsonify

from . import storage

DATA_DIR = storage.RUNTIME_DIR
USERS_FILE = storage.runtime_path("users.json")

# Reentrant lock: register_user() holds the lock and then calls _ensure_seed(),
# which also acquires it. A plain Lock would deadlock on that re-entry.
_lock = threading.RLock()

# Seed accounts created if no users file exists yet.
_SEED_USERS = [
    {"username": "admin", "password": "admin123", "role": "admin"},
    {"username": "customer", "password": "customer123", "role": "customer"},
]


def _hash_password(password, salt):
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def _read_users():
    # None only when the file truly doesn't exist yet. A corrupt file is
    # restored from backup (or raises) — it must never trigger reseeding,
    # which would overwrite every registered account.
    return storage.read_json(USERS_FILE, None)


def _write_users(users):
    storage.write_json(USERS_FILE, users)


def _ensure_seed():
    """Create the users file with seed accounts if it does not exist."""
    with _lock:
        users = _read_users()
        if users is not None:
            return users
        users = []
        for u in _SEED_USERS:
            salt = secrets.token_hex(8)
            users.append({
                "username": u["username"],
                "salt": salt,
                "password_hash": _hash_password(u["password"], salt),
                "role": u["role"],
            })
        _write_users(users)
        return users


def register_user(username, password, role="customer"):
    """Create a new user. Returns (ok, message)."""
    username = (username or "").strip()
    if not username or not password:
        return False, "Username and password are required."
    if role not in ("customer", "admin"):
        role = "customer"
    with _lock:
        users = _ensure_seed()
        if any(u["username"].lower() == username.lower() for u in users):
            return False, "Username already exists."
        salt = secrets.token_hex(8)
        users.append({
            "username": username,
            "salt": salt,
            "password_hash": _hash_password(password, salt),
            "role": role,
        })
        _write_users(users)
    return True, "Account created."


def verify_user(username, password):
    """Return the user record if credentials are valid, else None."""
    users = _ensure_seed()
    username = (username or "").strip()
    for u in users:
        if u["username"].lower() == username.lower():
            if _hash_password(password, u["salt"]) == u["password_hash"]:
                return {"username": u["username"], "role": u["role"]}
            return None
    return None


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------
def login_session(user):
    session["username"] = user["username"]
    session["role"] = user["role"]


def logout_session():
    session.pop("username", None)
    session.pop("role", None)


def current_user():
    if "username" in session:
        return {"username": session["username"], "role": session.get("role")}
    return None


def _wants_json():
    return request.path.startswith("/api/") or \
        request.accept_mimetypes.best == "application/json"


def login_required(role=None):
    """Decorator: require a logged-in user, optionally of a specific role."""
    def decorator(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            user = current_user()
            if user is None:
                if _wants_json():
                    return jsonify({"success": False, "error": "Login required"}), 401
                return redirect(url_for("login_page", next=request.path))
            if role and user["role"] != role:
                if _wants_json():
                    return jsonify({"success": False, "error": "Forbidden"}), 403
                return redirect(url_for("index"))
            return view(*args, **kwargs)
        return wrapped
    return decorator
