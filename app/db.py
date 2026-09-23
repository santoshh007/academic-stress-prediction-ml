"""
db.py — SQLite database helpers for storing anonymous survey responses.

Stores responses in a single SQLite file: app/collected_data/responses.db
"""

import os
import sqlite3
from datetime import datetime, timezone

# ---------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------
APP_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "collected_data")
DB_PATH  = os.path.join(DATA_DIR, "responses.db")

# Columns in the responses table (in order)
COLUMNS = [
    "timestamp",
    "age", "gender", "university", "program", "year", "cgpa",
    "study_hours", "attendance", "exam_count",
    "assignment_stress", "academic_performance",
    "sleep_hours", "social_media", "screen_time",
    "physical_activity", "part_time_job",
    "financial_stress", "social_support", "career_stress",
    "academic_life_satisfaction", "thought_break",
    "predicted_label",
]


def _ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def init_db():
    """Create the responses table if it doesn't exist yet."""
    _ensure_dir()
    conn = sqlite3.connect(DB_PATH)
    try:
        cols_sql = ",\n    ".join(f"{c} TEXT" for c in COLUMNS)
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                {cols_sql}
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def insert_response(user_input: dict, predicted_label: str) -> int:
    """
    Insert one anonymous response into the database.
    Returns the id of the newly inserted row.
    """
    _ensure_dir()
    init_db()

    record = dict(user_input)
    record["predicted_label"] = predicted_label
    record["timestamp"] = datetime.now(timezone.utc).isoformat()

    values = [str(record.get(col, "")) for col in COLUMNS]

    placeholders = ", ".join(["?"] * len(COLUMNS))
    cols_joined  = ", ".join(COLUMNS)

    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            f"INSERT INTO responses ({cols_joined}) VALUES ({placeholders})",
            values,
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def count_responses() -> int:
    """Return the total number of saved responses."""
    if not os.path.exists(DB_PATH):
        return 0
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM responses")
        return cur.fetchone()[0]
    finally:
        conn.close()