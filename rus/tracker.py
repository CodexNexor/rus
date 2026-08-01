"""
RUS Experience Tracker — SQLite database for logging ablation runs,
tracking success rates, and building the learning knowledge base.
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import Optional, List, Dict

from .config import DB_PATH


def _get_db_path() -> str:
    """Ensure the DB directory exists and return the path."""
    db_dir = os.path.dirname(DB_PATH)
    os.makedirs(db_dir, exist_ok=True)
    return DB_PATH


def _get_conn() -> sqlite3.Connection:
    """Get a connection to the experience database."""
    conn = sqlite3.connect(_get_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            num_prompts INTEGER,
            top_k_layers INTEGER,
            coefficient REAL,
            layers_modified TEXT,
            coefficients_used TEXT,
            refusal_before REAL,
            refusal_after REAL,
            compliance_before REAL,
            compliance_after REAL,
            quality_before REAL,
            quality_after REAL,
            export_path TEXT,
            duration_seconds REAL
        );

        CREATE TABLE IF NOT EXISTS layer_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            layer_idx INTEGER NOT NULL,
            refusal_score REAL,
            coefficient_applied REAL,
            projection_reduction REAL,
            FOREIGN KEY (run_id) REFERENCES runs(id)
        );

        CREATE TABLE IF NOT EXISTS model_insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_family TEXT NOT NULL,
            best_layers TEXT,
            best_coefficient REAL,
            avg_refusal_score_best REAL,
            num_runs INTEGER DEFAULT 1,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()


def log_run(
    model_name: str,
    num_prompts: int,
    top_k: int,
    coefficient: float,
    layers_modified: List[int],
    coefficients_used: List[float],
    comparison: Dict,
    export_path: Optional[str],
    duration_seconds: float,
    layer_refusal_scores: Dict[int, float],
):
    """Log a completed ablation run to the database."""
    init_db()
    conn = _get_conn()

    cursor = conn.execute(
        """INSERT INTO runs (
            model_name, num_prompts, top_k_layers, coefficient,
            layers_modified, coefficients_used,
            refusal_before, refusal_after,
            compliance_before, compliance_after,
            quality_before, quality_after,
            export_path, duration_seconds
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            model_name,
            num_prompts,
            top_k,
            coefficient,
            json.dumps(layers_modified),
            json.dumps(coefficients_used),
            comparison.get("refusal_rate_before", 0),
            comparison.get("refusal_rate_after", 0),
            comparison.get("compliance_before", 0),
            comparison.get("compliance_after", 0),
            comparison.get("quality_before", 0),
            comparison.get("quality_after", 0),
            export_path or "",
            duration_seconds,
        ),
    )
    run_id = cursor.lastrowid

    for layer_idx, score in layer_refusal_scores.items():
        conn.execute(
            """INSERT INTO layer_stats (run_id, layer_idx, refusal_score)
               VALUES (?, ?, ?)""",
            (run_id, layer_idx, score),
        )

    family = _extract_family(model_name)
    for layer_idx, score in layer_refusal_scores.items():
        if score >= 0.5:
            conn.execute(
                """INSERT INTO model_insights (model_family, best_layers, best_coefficient,
                   avg_refusal_score_best, num_runs)
                   VALUES (?, ?, ?, ?, 1)
                   ON CONFLICT(model_family) DO UPDATE SET
                   best_layers = json_insert(best_layers, '$[#]', ?),
                   avg_refusal_score_best = (avg_refusal_score_best * num_runs + ?) / (num_runs + 1),
                   num_runs = num_runs + 1,
                   last_updated = CURRENT_TIMESTAMP""",
                (family, json.dumps([layer_idx]), coefficient, score, layer_idx, score),
            )
            break

    conn.commit()
    conn.close()


def get_insights(model_family: str) -> Optional[Dict]:
    """Get learned insights for a model family."""
    init_db()
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM model_insights WHERE model_family = ? ORDER BY num_runs DESC LIMIT 1",
        (model_family,),
    ).fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


def get_recent_runs(limit: int = 5) -> List[Dict]:
    """Get the most recent ablation runs."""
    init_db()
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM runs ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _extract_family(model_name: str) -> str:
    """Extract model family from name. e.g. 'meta-llama/Meta-Llama-3-8B-Instruct' -> 'llama'"""
    name_lower = model_name.lower()
    if "llama" in name_lower:
        return "llama"
    if "mistral" in name_lower:
        return "mistral"
    if "qwen" in name_lower:
        return "qwen"
    if "gemma" in name_lower:
        return "gemma"
    if "phi" in name_lower:
        return "phi"
    if "falcon" in name_lower:
        return "falcon"
    if "deepseek" in name_lower:
        return "deepseek"
    if "gpt" in name_lower:
        return "gpt"
    return name_lower.split("/")[-1][:30]
