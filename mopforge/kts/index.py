"""SQLite metadata index for JSONL-backed Knowledge Training Stores."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from mopforge.kts.exceptions import LessonStoreError, LessonValidationError
from mopforge.kts.schema import KnowledgeLesson
from mopforge.kts.store import LessonStore


class LessonIndex:
    """SQLite metadata index for canonical JSONL lesson records.

    JSONL remains the source of truth. The index stores queryable metadata and
    pointers back to JSONL lines for fast counts and curriculum-style filtering.
    """

    COUNT_BY_FIELDS = {
        "domain",
        "skill",
        "subskill",
        "verification_status",
        "target_module",
    }

    def __init__(self, index_path: str | Path) -> None:
        """Open or create an index at ``index_path``."""

        self.index_path = Path(index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.create_schema()

    def create_schema(self) -> None:
        """Create the SQLite schema and indexes if they do not exist."""

        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS lessons(
                    id TEXT PRIMARY KEY,
                    domain TEXT NOT NULL,
                    skill TEXT NOT NULL,
                    subskill TEXT,
                    difficulty INTEGER NOT NULL,
                    verification_status TEXT,
                    verification_type TEXT,
                    jsonl_offset INTEGER,
                    jsonl_line INTEGER,
                    created_at TEXT,
                    source_path TEXT
                );

                CREATE TABLE IF NOT EXISTS lesson_modules(
                    lesson_id TEXT NOT NULL,
                    module TEXT NOT NULL,
                    PRIMARY KEY (lesson_id, module),
                    FOREIGN KEY (lesson_id) REFERENCES lessons(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS lesson_metadata(
                    lesson_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT,
                    PRIMARY KEY (lesson_id, key),
                    FOREIGN KEY (lesson_id) REFERENCES lessons(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_lessons_domain ON lessons(domain);
                CREATE INDEX IF NOT EXISTS idx_lessons_skill ON lessons(skill);
                CREATE INDEX IF NOT EXISTS idx_lessons_subskill ON lessons(subskill);
                CREATE INDEX IF NOT EXISTS idx_lessons_difficulty ON lessons(difficulty);
                CREATE INDEX IF NOT EXISTS idx_lessons_verification_status
                    ON lessons(verification_status);
                CREATE INDEX IF NOT EXISTS idx_lessons_verification_type
                    ON lessons(verification_type);
                CREATE INDEX IF NOT EXISTS idx_modules_module
                    ON lesson_modules(module);
                CREATE INDEX IF NOT EXISTS idx_metadata_key_value
                    ON lesson_metadata(key, value);
                """
            )

    def clear(self) -> None:
        """Remove all indexed lesson metadata."""

        with self._connect() as conn:
            conn.execute("DELETE FROM lesson_metadata")
            conn.execute("DELETE FROM lesson_modules")
            conn.execute("DELETE FROM lessons")

    def index_lesson(
        self,
        lesson: KnowledgeLesson,
        *,
        jsonl_line: int | None = None,
        jsonl_offset: int | None = None,
        source_path: str | Path | None = None,
    ) -> None:
        """Insert or replace metadata for one validated lesson."""

        lesson.validate()
        with self._connect() as conn:
            self._index_lesson_with_conn(
                conn,
                lesson,
                jsonl_line=jsonl_line,
                jsonl_offset=jsonl_offset,
                source_path=source_path,
            )

    def rebuild_from_store(self, store: LessonStore | str | Path) -> int:
        """Clear and rebuild the index from a JSONL store.

        Returns:
            Number of lessons indexed.
        """

        lesson_path = store.path if isinstance(store, LessonStore) else Path(store)
        lesson_path = Path(lesson_path)
        self.clear()
        if not lesson_path.exists():
            return 0

        indexed = 0
        with lesson_path.open("r", encoding="utf-8") as file, self._connect() as conn:
            line_number = 0
            while True:
                offset = file.tell()
                line = file.readline()
                if not line:
                    break
                line_number += 1
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    data = json.loads(stripped)
                    lesson = KnowledgeLesson.from_dict(data)
                except json.JSONDecodeError as exc:
                    raise LessonStoreError(
                        f"Invalid JSON in {lesson_path} at line {line_number}: "
                        f"{exc.msg}."
                    ) from exc
                except LessonValidationError as exc:
                    raise LessonStoreError(
                        f"Invalid lesson in {lesson_path} at line {line_number}: {exc}"
                    ) from exc

                self._index_lesson_with_conn(
                    conn,
                    lesson,
                    jsonl_line=line_number,
                    jsonl_offset=offset,
                    source_path=lesson_path,
                )
                indexed += 1
        return indexed

    def query_ids(self, **filters: Any) -> list[str]:
        """Return lesson IDs matching index filters."""

        rows = self.query(**filters)
        return [str(row["id"]) for row in rows]

    def query(self, **filters: Any) -> list[dict[str, Any]]:
        """Return small metadata dictionaries matching index filters."""

        limit = filters.pop("limit", None)
        where_sql, params = self._build_where(filters)
        sql = (
            "SELECT id, domain, skill, subskill, difficulty, "
            "verification_status, verification_type, jsonl_offset, jsonl_line, "
            "created_at, source_path FROM lessons "
            f"{where_sql} ORDER BY jsonl_line IS NULL, jsonl_line, id"
        )
        if limit is not None:
            if type(limit) is not int or limit < 0:
                raise ValueError("limit must be a non-negative integer.")
            sql += " LIMIT ?"
            params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def count(self, **filters: Any) -> int:
        """Return number of lessons matching index filters."""

        where_sql, params = self._build_where(filters)
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) AS count FROM lessons {where_sql}", params
            ).fetchone()
        return int(row["count"])

    def count_by(self, field: str, **filters: Any) -> dict[str | None, int]:
        """Count matching lessons grouped by a supported field."""

        if field not in self.COUNT_BY_FIELDS:
            valid = ", ".join(sorted(self.COUNT_BY_FIELDS))
            raise ValueError(f"field must be one of: {valid}.")

        where_sql, params = self._build_where(filters)
        if field == "target_module":
            sql = (
                "SELECT lesson_modules.module AS key, COUNT(DISTINCT lessons.id) AS count "
                "FROM lessons JOIN lesson_modules ON lesson_modules.lesson_id = lessons.id "
                f"{where_sql} GROUP BY lesson_modules.module ORDER BY lesson_modules.module"
            )
        else:
            sql = (
                f"SELECT {field} AS key, COUNT(*) AS count FROM lessons "
                f"{where_sql} GROUP BY {field} ORDER BY {field}"
            )

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return {row["key"]: int(row["count"]) for row in rows}

    def export_query_json(self, path: str | Path, **query_kwargs: Any) -> Path:
        """Write query results to JSON and return the output path."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.query(**query_kwargs), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output_path

    def rows_by_ids(self, lesson_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Return indexed lesson rows keyed by lesson ID."""

        if not lesson_ids:
            return {}
        placeholders = ", ".join("?" for _ in lesson_ids)
        sql = (
            "SELECT id, domain, skill, subskill, difficulty, "
            "verification_status, verification_type, jsonl_offset, jsonl_line, "
            "created_at, source_path FROM lessons "
            f"WHERE id IN ({placeholders})"
        )
        with self._connect() as conn:
            rows = conn.execute(sql, lesson_ids).fetchall()
        return {str(row["id"]): dict(row) for row in rows}

    def modules_by_ids(self, lesson_ids: list[str]) -> dict[str, list[str]]:
        """Return target modules keyed by lesson ID."""

        if not lesson_ids:
            return {}
        placeholders = ", ".join("?" for _ in lesson_ids)
        sql = (
            "SELECT lesson_id, module FROM lesson_modules "
            f"WHERE lesson_id IN ({placeholders}) ORDER BY lesson_id, module"
        )
        modules: dict[str, list[str]] = {lesson_id: [] for lesson_id in lesson_ids}
        with self._connect() as conn:
            rows = conn.execute(sql, lesson_ids).fetchall()
        for row in rows:
            modules.setdefault(str(row["lesson_id"]), []).append(str(row["module"]))
        return modules

    def _index_lesson_with_conn(
        self,
        conn: sqlite3.Connection,
        lesson: KnowledgeLesson,
        *,
        jsonl_line: int | None,
        jsonl_offset: int | None,
        source_path: str | Path | None,
    ) -> None:
        conn.execute(
            """
            INSERT OR REPLACE INTO lessons(
                id, domain, skill, subskill, difficulty,
                verification_status, verification_type,
                jsonl_offset, jsonl_line, created_at, source_path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                lesson.id,
                lesson.domain,
                lesson.skill,
                lesson.subskill,
                lesson.difficulty,
                lesson.verification.get("status"),
                lesson.verification.get("type"),
                jsonl_offset,
                jsonl_line,
                lesson.created_at,
                str(source_path) if source_path is not None else None,
            ),
        )
        conn.execute("DELETE FROM lesson_modules WHERE lesson_id = ?", (lesson.id,))
        conn.execute("DELETE FROM lesson_metadata WHERE lesson_id = ?", (lesson.id,))
        conn.executemany(
            "INSERT OR REPLACE INTO lesson_modules(lesson_id, module) VALUES (?, ?)",
            [(lesson.id, module) for module in lesson.target_modules],
        )
        conn.executemany(
            "INSERT OR REPLACE INTO lesson_metadata(lesson_id, key, value) VALUES (?, ?, ?)",
            [
                (lesson.id, key, _metadata_value(value))
                for key, value in sorted(lesson.metadata.items())
            ],
        )

    def _build_where(self, filters: dict[str, Any]) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []

        field_map = {
            "domain": "domain",
            "skill": "skill",
            "subskill": "subskill",
            "verification_status": "verification_status",
            "verification_type": "verification_type",
        }
        for filter_name, column_name in field_map.items():
            value = filters.pop(filter_name, None)
            if value is not None:
                clauses.append(f"{column_name} = ?")
                params.append(value)

        min_difficulty = filters.pop(
            "difficulty_min", filters.pop("min_difficulty", None)
        )
        max_difficulty = filters.pop(
            "difficulty_max", filters.pop("max_difficulty", None)
        )
        if min_difficulty is not None:
            clauses.append("difficulty >= ?")
            params.append(min_difficulty)
        if max_difficulty is not None:
            clauses.append("difficulty <= ?")
            params.append(max_difficulty)

        target_modules = filters.pop("target_modules", None)
        module_match = filters.pop("module_match", "any")
        if target_modules is not None:
            modules = _normalize_modules(target_modules)
            if module_match not in {"any", "all"}:
                raise ValueError("module_match must be either 'any' or 'all'.")
            if module_match == "any":
                placeholders = ", ".join("?" for _ in modules)
                clauses.append(
                    "EXISTS (SELECT 1 FROM lesson_modules lm "
                    "WHERE lm.lesson_id = lessons.id "
                    f"AND lm.module IN ({placeholders}))"
                )
                params.extend(modules)
            else:
                for module in modules:
                    clauses.append(
                        "EXISTS (SELECT 1 FROM lesson_modules lm "
                        "WHERE lm.lesson_id = lessons.id AND lm.module = ?)"
                    )
                    params.append(module)

        metadata = filters.pop("metadata", filters.pop("metadata_contains", None))
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be a dictionary.")
            for key, value in sorted(metadata.items()):
                clauses.append(
                    "EXISTS (SELECT 1 FROM lesson_metadata md "
                    "WHERE md.lesson_id = lessons.id AND md.key = ? AND md.value = ?)"
                )
                params.extend([key, _metadata_value(value)])

        if filters:
            unknown = ", ".join(sorted(filters))
            raise ValueError(f"Unknown query filter(s): {unknown}.")

        if not clauses:
            return "", params
        return "WHERE " + " AND ".join(clauses), params

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.index_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn


def _metadata_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return str(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _normalize_modules(value: Any) -> list[str]:
    if isinstance(value, str):
        modules = [value]
    else:
        modules = list(value)
    if not modules or not all(isinstance(module, str) and module.strip() for module in modules):
        raise ValueError("target_modules must contain non-empty strings.")
    seen = set()
    normalized = []
    for module in modules:
        if module not in seen:
            normalized.append(module)
            seen.add(module)
    return normalized
