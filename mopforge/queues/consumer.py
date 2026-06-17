"""Tiny local queue consumer."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from mopforge.queues.store import TrainingQueueStore


def consume_queue_once(
    queue_store: TrainingQueueStore,
    *,
    module: str | None = None,
    run_registry_root: str = "runs",
    dry_run: bool = False,
    artifact_manager=None,
) -> dict:
    """Process or preview one local queue item.

    Dry runs are non-mutating: they return the next pending item without
    claiming it. Non-dry-runs claim the item, write minimal run metadata, and
    mark it done. This is queue plumbing only, not real training.
    """

    if dry_run:
        items = queue_store.list_items(status="pending", module=module, limit=1)
        if not items:
            return _empty_result(dry_run=True)
        item = items[0]
        return {
            "item_id": item.item_id,
            "module": item.module,
            "lesson_id": item.lesson_id,
            "status": item.status,
            "attempts": item.attempts,
            "run_id": item.run_id,
            "dry_run": True,
            "message": "preview only; item was not claimed",
        }

    item = queue_store.claim_next(module=module)
    if item is None:
        return _empty_result(dry_run=False)

    run_id = _make_queue_run_id(item.item_id)
    run_dir = Path(run_registry_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = run_dir / "queue_item.json"
    processed_at = _now()
    metadata = {
        "item_id": item.item_id,
        "module": item.module,
        "lesson_id": item.lesson_id,
        "queue_status": "done",
        "attempts": item.attempts,
        "processed_at": processed_at,
        "dry_run": False,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    queue_store.mark_done(
        item.item_id,
        run_id=run_id,
        metadata={
            "consumer": "consume_queue_once",
            "processed_at": processed_at,
            "queue_item_json": str(metadata_path),
        },
    )
    artifact_id = None
    if artifact_manager is not None:
        from mopforge.artifacts import ArtifactRecord

        artifact = artifact_manager.register(
            ArtifactRecord(
                artifact_id=f"queue-item-{run_id}",
                kind="queue_item",
                path=str(metadata_path),
                run_id=run_id,
                queue_item_id=item.item_id,
                module=item.module,
                metadata={"lesson_id": item.lesson_id},
            )
        )
        artifact_id = artifact.artifact_id
    updated = queue_store.get(item.item_id)
    return {
        "item_id": item.item_id,
        "module": item.module,
        "lesson_id": item.lesson_id,
        "status": updated.status if updated is not None else "done",
        "attempts": updated.attempts if updated is not None else item.attempts,
        "run_id": run_id,
        "artifact_id": artifact_id,
        "dry_run": False,
        "message": "item marked done by local smoke consumer",
    }


def _empty_result(*, dry_run: bool) -> dict:
    return {
        "item_id": None,
        "module": None,
        "lesson_id": None,
        "status": "empty",
        "attempts": 0,
        "run_id": None,
        "dry_run": dry_run,
        "message": "no pending queue item",
    }


def _make_queue_run_id(item_id: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_item_id = "".join(
        ch.lower() if ch.isalnum() else "-" for ch in item_id
    ).strip("-")
    return f"{timestamp}-queue-{safe_item_id or 'item'}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
