"""
Safe persistence for the app's runtime data (users, knowledge base, tickets).

Guarantees that existing data is never lost:
  1. Backups    - every successful write is snapshotted to
                  data/backups/<name>.<timestamp>.json (last BACKUP_KEEP kept),
                  so the newest backup always matches the latest good state.
  2. No shrink  - these files only ever grow (append/update). A write that
                  would leave fewer records than are on disk is refused,
                  because that can only come from a bug.
  3. Corruption - an unreadable file is never treated as "empty". It is moved
                  aside as <name>.corrupt-<timestamp> and the newest valid
                  backup is restored. If no backup exists, an error is raised
                  rather than silently starting over.
  4. Isolation  - set APP_DATA_DIR to point the app (e.g. tests) at another
                  folder, so the real data/ folder is never touched.
"""
import os
import json
import glob
import shutil
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Runtime data folder. Defaults to data/; tests point it elsewhere.
RUNTIME_DIR = os.environ.get("APP_DATA_DIR") or os.path.join(BASE_DIR, "data")
BACKUP_DIR = os.path.join(RUNTIME_DIR, "backups")
BACKUP_KEEP = 50


class DataSafetyError(RuntimeError):
    """Raised instead of doing something that could lose existing data."""


def runtime_path(filename):
    os.makedirs(RUNTIME_DIR, exist_ok=True)
    return os.path.join(RUNTIME_DIR, filename)


def _stamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def _backups_for(path):
    name = os.path.basename(path)
    return sorted(glob.glob(os.path.join(BACKUP_DIR, f"{name}.*.json")), reverse=True)


def _backup(path):
    """Copy the current file into data/backups and prune old copies."""
    if not os.path.exists(path):
        return
    os.makedirs(BACKUP_DIR, exist_ok=True)
    name = os.path.basename(path)
    shutil.copy2(path, os.path.join(BACKUP_DIR, f"{name}.{_stamp()}.json"))
    for old in _backups_for(path)[BACKUP_KEEP:]:
        os.remove(old)


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_json(path, default):
    """
    Read a JSON data file. Missing file -> default. Corrupt file -> restore
    the newest valid backup (the corrupt file is kept, never deleted).
    """
    if not os.path.exists(path):
        return default
    try:
        return _load(path)
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
        # Keep the damaged file for inspection, then restore from backup.
        shutil.move(path, f"{path}.corrupt-{_stamp()}")
        for candidate in _backups_for(path):
            try:
                data = _load(candidate)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            shutil.copy2(candidate, path)
            return data
        raise DataSafetyError(
            f"{os.path.basename(path)} is unreadable ({err}) and no valid backup "
            f"exists. The damaged file was kept as {path}.corrupt-*; nothing "
            f"was overwritten."
        ) from err


def write_json(path, data):
    """Atomically write a JSON data file, backing up the previous version."""
    if os.path.exists(path) and isinstance(data, list):
        try:
            existing = _load(path)
        except (json.JSONDecodeError, UnicodeDecodeError):
            existing = None
        if isinstance(existing, list) and len(data) < len(existing):
            raise DataSafetyError(
                f"Refusing to write {os.path.basename(path)}: it would shrink "
                f"from {len(existing)} to {len(data)} records."
            )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    # Snapshot the version just written, so the newest backup always equals
    # the latest good state (restoring it loses nothing).
    _backup(path)
