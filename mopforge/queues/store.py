"""SQLite-backed local training queue store."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from mopforge.queues.schema import TrainingQueueItem


class TrainingQueueStore:
    """File-backed local queue for module-targeted training items."""

    def __init__(self, path: str | Path) -> None:
        """Open or create a queue database."""

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.create_schema()

    def create_schema(self) -> None:
        """Create queue tables and indexes if they do not exist."""

        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS training_queue(
                    item_id TEXT PRIMARY KEY,
                    module TEXT NOT NULL,
                    lesson_id TEXT NOT NULL,
                    priority REAL NOT NULL,
                    status TEXT NOT NULL,
                    source TEXT,
                    run_id TEXT,
                    attempts INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata_json TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_training_queue_module
                    ON training_queue(module);
                CREATE INDEX IF NOT EXISTS idx_training_queue_status
                    ON training_queue(status);
                CREATE INDEX IF NOT EXISTS idx_training_queue_priority
                    ON training_queue(priority);
                CREATE INDEX IF NOT EXISTS idx_training_queue_lesson_id
                    ON training_queue(lesson_id);
                """
            )

    def add_item(self, item: TrainingQueueItem) -> None:
        """Insert one queue item."""

        item.validate()
        with self._connect() as conn:
            self._insert_item(conn, item)

    def add_many(self, items: Iterable[TrainingQueueItem]) -> int:
        """Insert many queue items and return the number inserted."""

        inserted = 0
        with self._connect() as conn:
            for item in items:
                item.validate()
                self._insert_item(conn, item)
                inserted += 1
        return inserted

    def get(self, item_id: str) -> TrainingQueueItem | None:
        """Return one queue item by ID."""

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT item_id, module, lesson_id, priority, status, source,
                       run_id, attempts, created_at, updated_at, metadata_json
                FROM training_queue
                WHERE item_id = ?
                """,
                (item_id,),
            ).fetchone()
        return _row_to_item(row) if row is not None else None

    def list_items(
        self,
        status: str | None = None,
        module: str | None = None,
        limit: int | None = None,
    ) -> list[TrainingQueueItem]:
        """List queue items, optionally filtered by status and module."""

        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if module is not None:
            clauses.append("module = ?")
            params.append(module)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (
            "SELECT item_id, module, lesson_id, priority, status, source, "
            "run_id, attempts, created_at, updated_at, metadata_json "
            f"FROM training_queue {where_sql} "
            "ORDER BY priority DESC, created_at ASC, item_id ASC"
        )
        if limit is not None:
            if type(limit) is not int or limit < 0:
                raise ValueError("limit must be a non-negative integer.")
            sql += " LIMIT ?"
            params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_item(row) for row in rows]

    def claim_next(self, module: str | None = None) -> TrainingQueueItem | None:
        """Claim the highest-priority pending item and mark it running."""

        clauses = ["status = ?"]
        params: list[Any] = ["pending"]
        if module is not None:
            clauses.append("module = ?")
            params.append(module)
        now = _now()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT item_id
                FROM training_queue
                WHERE """
                + " AND ".join(clauses)
                + """
                ORDER BY priority DESC, created_at ASC, item_id ASC
                LIMIT 1
                """,
                params,
            ).fetchone()
            if row is None:
                return None
            item_id = str(row["item_id"])
            conn.execute(
                """
                UPDATE training_queue
                SET status = ?, attempts = attempts + 1, updated_at = ?
                WHERE item_id = ?
                """,
                ("running", now, item_id),
            )
        return self.get(item_id)

    def mark_done(
        self,
        item_id: str,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Mark a queue item done."""

        self._update_status(item_id, "done", run_id=run_id, metadata=metadata)

    def mark_failed(
        self,
        item_id: str,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Mark a queue item failed."""

        merged = dict(metadata or {})
        if error is not None:
            merged["error"] = error
        self._update_status(item_id, "failed", metadata=merged)

    def mark_skipped(self, item_id: str, reason: str | None = None) -> None:
        """Mark a queue item skipped."""

        metadata = {"skip_reason": reason} if reason is not None else None
        self._update_status(item_id, "skipped", metadata=metadata)

    def count(
        self,
        status: str | None = None,
        module: str | None = None,
    ) -> int:
        """Count queue items, optionally filtered by status and module."""

        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if module is not None:
            clauses.append("module = ?")
            params.append(module)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) AS count FROM training_queue {where_sql}",
                params,
            ).fetchone()
        return int(row["count"])

    def counts_by_status(self) -> dict[str, int]:
        """Return queue item counts grouped by status."""

        return self._count_by("status")

    def counts_by_module(self) -> dict[str, int]:
        """Return queue item counts grouped by module."""

        return self._count_by("module")

    def export_json(self, path: str | Path) -> Path:
        """Export queue items and grouped counts to JSON."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "items": [item.to_dict() for item in self.list_items()],
            "counts_by_status": self.counts_by_status(),
            "counts_by_module": self.counts_by_module(),
        }
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output_path

    def _insert_item(self, conn: sqlite3.Connection, item: TrainingQueueItem) -> None:
        conn.execute(
            """
            INSERT INTO training_queue(
                item_id, module, lesson_id, priority, status, source, run_id,
                attempts, created_at, updated_at, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.item_id,
                item.module,
                item.lesson_id,
                float(item.priority),
                item.status,
                item.source,
                item.run_id,
                item.attempts,
                item.created_at,
                item.updated_at,
                json.dumps(item.metadata, sort_keys=True, separators=(",", ":")),
            ),
        )

    def _update_status(
        self,
        item_id: str,
        status: str,
        *,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        existing = self.get(item_id)
        if existing is None:
            raise KeyError(f"Queue item not found: {item_id}")
        merged_metadata = dict(existing.metadata)
        if metadata:
            merged_metadata.update(metadata)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE training_queue
                SET status = ?, run_id = COALESCE(?, run_id),
                    updated_at = ?, metadata_json = ?
                WHERE item_id = ?
                """,
                (
                    status,
                    run_id,
                    _now(),
                    json.dumps(
                        merged_metadata,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    item_id,
                ),
            )

    def _count_by(self, field: str) -> dict[str, int]:
        if field not in {"status", "module"}:
            raise ValueError("field must be status or module.")
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {field} AS key, COUNT(*) AS count "
                "FROM training_queue GROUP BY "
                f"{field} ORDER BY {field}"
            ).fetchall()
        return {str(row["key"]): int(row["count"]) for row in rows}

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn


def _row_to_item(row: sqlite3.Row) -> TrainingQueueItem:
    return TrainingQueueItem(
        item_id=str(row["item_id"]),
        module=str(row["module"]),
        lesson_id=str(row["lesson_id"]),
        priority=float(row["priority"]),
        status=str(row["status"]),
        source=str(row["source"] or "curriculum"),
        run_id=row["run_id"],
        attempts=int(row["attempts"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        metadata=json.loads(row["metadata_json"] or "{}"),
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
