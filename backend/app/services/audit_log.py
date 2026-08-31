"""
SQLite-backed audit trail. This is what the PS's evaluation criterion means by
"an auditable execution summary" — keep it simple and queryable, not a log file
nobody looks at during judging.
"""
import json
import sqlite3
import uuid
from datetime import datetime, timezone

from app.config import AUDIT_DB_PATH


def init_db() -> None:
    with sqlite3.connect(AUDIT_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                query TEXT NOT NULL,
                input_config TEXT NOT NULL,
                task TEXT NOT NULL,
                tools_used TEXT NOT NULL,
                confidence REAL NOT NULL,
                execution_trace TEXT NOT NULL,
                answer TEXT NOT NULL
            )
            """
        )


def log_execution(
    query: str,
    input_config: str,
    task: str,
    tools_used: list[str],
    confidence: float,
    execution_trace: list[dict],
    answer: str,
) -> str:
    report_id = str(uuid.uuid4())
    with sqlite3.connect(AUDIT_DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO audit_log
            (id, created_at, query, input_config, task, tools_used, confidence, execution_trace, answer)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report_id,
                datetime.now(timezone.utc).isoformat(),
                query,
                input_config,
                task,
                json.dumps(tools_used),
                confidence,
                json.dumps(execution_trace),
                answer,
            ),
        )
    return report_id


def get_execution(report_id: str) -> dict | None:
    with sqlite3.connect(AUDIT_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM audit_log WHERE id = ?", (report_id,)).fetchone()
        return dict(row) if row else None
