"""SQLite-backed store for per-lesson feedback events."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from mopforge.feedback.schema import LessonFeedbackRecord


class LessonFeedbackStore:
    """File-backed SQLite store for lesson feedback and summaries.

    Every feedback event increments ``attempts`` in the summary table. Records
    with ``passed=True`` increment passes, records with ``passed=False``
    increment failures, and records with ``passed=None`` are counted as attempts
    only.
    """

    def __init__(self, path: str | Path) -> None:
        """Open or create a lesson feedback database."""

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.create_schema()

    def create_schema(self) -> None:
        """Create feedback tables and indexes if they do not exist."""

        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS lesson_feedback(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lesson_id TEXT NOT NULL,
                    run_id TEXT,
                    model_type TEXT,
                    curriculum_strategy TEXT,
                    passed INTEGER,
                    failure_type TEXT,
                    loss REAL,
                    generated INTEGER NOT NULL DEFAULT 0,
                    timestamp TEXT NOT NULL,
                    metadata_json TEXT
                );

                CREATE TABLE IF NOT EXISTS lesson_feedback_summary(
                    lesson_id TEXT PRIMARY KEY,
                    attempts INTEGER NOT NULL,
                    passes INTEGER NOT NULL,
                    failures INTEGER NOT NULL,
                    avg_loss REAL,
                    last_failure_type TEXT,
                    last_seen TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_feedback_lesson_id
                    ON lesson_feedback(lesson_id);
                CREATE INDEX IF NOT EXISTS idx_feedback_failure_type
                    ON lesson_feedback(failure_type);
                CREATE INDEX IF NOT EXISTS idx_feedback_timestamp
                    ON lesson_feedback(timestamp);
                """
            )

    def add_feedback(self, record: LessonFeedbackRecord) -> int:
        """Add one feedback event and return its row ID."""

        record.validate()
        timestamp = record.timestamp or datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            row_id = self._insert_record(conn, record, timestamp)
            self._refresh_summary(conn, record.lesson_id)
        return row_id

    def add_many(self, records: Iterable[LessonFeedbackRecord]) -> int:
        """Add many feedback events and return the number inserted."""

        inserted = 0
        changed_lesson_ids: set[str] = set()
        with self._connect() as conn:
            for record in records:
                record.validate()
                timestamp = record.timestamp or datetime.now(timezone.utc).isoformat()
                self._insert_record(conn, record, timestamp)
                changed_lesson_ids.add(record.lesson_id)
                inserted += 1
            for lesson_id in changed_lesson_ids:
                self._refresh_summary(conn, lesson_id)
        return inserted

    def summary_for_lesson(self, lesson_id: str) -> dict[str, Any]:
        """Return summary metrics for one lesson ID."""

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT lesson_id, attempts, passes, failures, avg_loss,
                       last_failure_type, last_seen
                FROM lesson_feedback_summary
                WHERE lesson_id = ?
                """,
                (lesson_id,),
            ).fetchone()
        if row is None:
            return {
                "lesson_id": lesson_id,
                "attempts": 0,
                "passes": 0,
                "failures": 0,
                "avg_loss": None,
                "last_failure_type": None,
                "last_seen": None,
            }
        return _summary_row_to_dict(row)

    def summaries_for_lessons(self, lesson_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Return summary metrics keyed by lesson ID."""

        return {lesson_id: self.summary_for_lesson(lesson_id) for lesson_id in lesson_ids}

    def count(self) -> int:
        """Return number of stored feedback events."""

        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM lesson_feedback").fetchone()
        return int(row["count"])

    def failure_counts_by_type(self) -> dict[str, int]:
        """Return failed feedback counts grouped by failure type."""

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT COALESCE(failure_type, 'unknown') AS failure_type,
                       COUNT(*) AS count
                FROM lesson_feedback
                WHERE passed = 0
                GROUP BY COALESCE(failure_type, 'unknown')
                ORDER BY failure_type
                """
            ).fetchall()
        return {str(row["failure_type"]): int(row["count"]) for row in rows}

    def export_json(self, path: str | Path) -> Path:
        """Export feedback events and summaries to JSON."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            feedback_rows = conn.execute(
                """
                SELECT lesson_id, run_id, model_type, curriculum_strategy, passed,
                       failure_type, loss, generated, timestamp, metadata_json
                FROM lesson_feedback
                ORDER BY id
                """
            ).fetchall()
            summary_rows = conn.execute(
                """
                SELECT lesson_id, attempts, passes, failures, avg_loss,
                       last_failure_type, last_seen
                FROM lesson_feedback_summary
                ORDER BY lesson_id
                """
            ).fetchall()
        payload = {
            "feedback": [_feedback_row_to_dict(row) for row in feedback_rows],
            "summaries": [_summary_row_to_dict(row) for row in summary_rows],
        }
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output_path

    def _insert_record(
        self,
        conn: sqlite3.Connection,
        record: LessonFeedbackRecord,
        timestamp: str,
    ) -> int:
        metadata_json = json.dumps(
            record.metadata,
            sort_keys=True,
            separators=(",", ":"),
        )
        cursor = conn.execute(
            """
            INSERT INTO lesson_feedback(
                lesson_id, run_id, model_type, curriculum_strategy,
                passed, failure_type, loss, generated, timestamp, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.lesson_id,
                record.run_id,
                record.model_type,
                record.curriculum_strategy,
                _bool_to_int_or_none(record.passed),
                record.failure_type,
                float(record.loss) if record.loss is not None else None,
                1 if record.generated else 0,
                timestamp,
                metadata_json,
            ),
        )
        return int(cursor.lastrowid)

    def _refresh_summary(self, conn: sqlite3.Connection, lesson_id: str) -> None:
        stats = conn.execute(
            """
            SELECT
                COUNT(*) AS attempts,
                SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END) AS passes,
                SUM(CASE WHEN passed = 0 THEN 1 ELSE 0 END) AS failures,
                AVG(loss) AS avg_loss,
                MAX(timestamp) AS last_seen
            FROM lesson_feedback
            WHERE lesson_id = ?
            """,
            (lesson_id,),
        ).fetchone()
        last_failure = conn.execute(
            """
            SELECT failure_type
            FROM lesson_feedback
            WHERE lesson_id = ? AND passed = 0
            ORDER BY timestamp DESC, id DESC
            LIMIT 1
            """,
            (lesson_id,),
        ).fetchone()
        conn.execute(
            """
            INSERT INTO lesson_feedback_summary(
                lesson_id, attempts, passes, failures, avg_loss,
                last_failure_type, last_seen
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(lesson_id) DO UPDATE SET
                attempts = excluded.attempts,
                passes = excluded.passes,
                failures = excluded.failures,
                avg_loss = excluded.avg_loss,
                last_failure_type = excluded.last_failure_type,
                last_seen = excluded.last_seen
            """,
            (
                lesson_id,
                int(stats["attempts"]),
                int(stats["passes"] or 0),
                int(stats["failures"] or 0),
                stats["avg_loss"],
                last_failure["failure_type"] if last_failure is not None else None,
                stats["last_seen"],
            ),
        )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn


def feedback_records_from_generation_eval(
    results: list[dict[str, Any]],
    *,
    run_id: str | None = None,
    model_type: str | None = None,
    curriculum_strategy: str | None = None,
) -> list[LessonFeedbackRecord]:
    """Convert generated-code evaluation results into feedback records."""

    records: list[LessonFeedbackRecord] = []
    for result in _iter_generation_results(results):
        lesson_id = result.get("lesson_id")
        if not isinstance(lesson_id, str) or not lesson_id.strip():
            continue
        result_model_type = result.get("_model_type") or model_type
        metadata = {
            key: result[key]
            for key in ("exit_code", "timeout", "target_modules")
            if key in result
        }
        if "routing" in result:
            metadata["routing"] = result["routing"]
        records.append(
            LessonFeedbackRecord(
                lesson_id=lesson_id,
                run_id=run_id,
                model_type=result_model_type,
                curriculum_strategy=curriculum_strategy,
                passed=result.get("passed"),
                failure_type=result.get("failure_type"),
                generated=True,
                metadata=metadata,
            )
        )
    return records


def feedback_records_from_run_record(record: Any) -> list[LessonFeedbackRecord]:
    """Return per-lesson feedback from a run record when available.

    Goal 11 run records store aggregate metrics only, so this currently returns
    an empty list unless future records add per-lesson feedback metadata.
    """

    _ = record
    return []


def _iter_generation_results(results: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for item in results:
        nested = item.get("results")
        if isinstance(nested, list):
            group_model = item.get("model")
            routing = item.get("routing")
            for result in nested:
                if isinstance(result, dict):
                    enriched = dict(result)
                    if group_model is not None:
                        enriched["_model_type"] = str(group_model)
                    if routing is not None:
                        enriched["routing"] = routing
                    yield enriched
        elif isinstance(item, dict):
            yield item


def _feedback_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "lesson_id": row["lesson_id"],
        "run_id": row["run_id"],
        "model_type": row["model_type"],
        "curriculum_strategy": row["curriculum_strategy"],
        "passed": _int_to_bool_or_none(row["passed"]),
        "failure_type": row["failure_type"],
        "loss": row["loss"],
        "generated": bool(row["generated"]),
        "timestamp": row["timestamp"],
        "metadata": json.loads(row["metadata_json"] or "{}"),
    }


def _summary_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "lesson_id": row["lesson_id"],
        "attempts": int(row["attempts"]),
        "passes": int(row["passes"]),
        "failures": int(row["failures"]),
        "avg_loss": row["avg_loss"],
        "last_failure_type": row["last_failure_type"],
        "last_seen": row["last_seen"],
    }


def _bool_to_int_or_none(value: bool | None) -> int | None:
    if value is None:
        return None
    return 1 if value else 0


def _int_to_bool_or_none(value: int | None) -> bool | None:
    if value is None:
        return None
    return bool(value)
