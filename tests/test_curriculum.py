"""Tests for deterministic curriculum scheduling."""

from __future__ import annotations

import json

from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.curriculum import CurriculumConfig, CurriculumScheduler
from mopforge.kts import IndexedLessonStore
from mopforge.repair import (
    build_repair_lesson_from_failure,
    failure_record_from_generation_result,
)


def build_store(tmp_path) -> IndexedLessonStore:
    store = IndexedLessonStore(tmp_path / "lessons.jsonl", tmp_path / "lessons.sqlite")
    lessons = generate_coding_bugfix_lessons(count_per_category=1, verify=False)
    for lesson in lessons:
        store.add(lesson)

    source = lessons[0]
    result = {
        "lesson_id": source.id,
        "passed": False,
        "failure_type": "syntax_error",
        "generated_text": "bad code",
        "candidate_code": "bad code",
        "exit_code": 1,
        "timeout": False,
        "target_modules": source.target_modules,
    }
    failure = failure_record_from_generation_result(result, source)
    assert failure is not None
    store.add(build_repair_lesson_from_failure(failure))
    return store


def test_curriculum_config_defaults() -> None:
    config = CurriculumConfig()

    assert config.strategy == "balanced"
    assert config.batch_size == 8
    assert config.seed == 123
    assert config.include_repair is True
    assert config.repair_boost == 1.0


def test_sequential_strategy_is_deterministic(tmp_path) -> None:
    scheduler = CurriculumScheduler(indexed_store=build_store(tmp_path))

    first = scheduler.build_plan(CurriculumConfig(strategy="sequential"))
    second = scheduler.build_plan(CurriculumConfig(strategy="sequential"))

    assert first.lesson_ids == second.lesson_ids
    assert first.lesson_ids == sorted(first.lesson_ids)


def test_shuffled_strategy_is_deterministic_with_seed(tmp_path) -> None:
    scheduler = CurriculumScheduler(indexed_store=build_store(tmp_path))
    config = CurriculumConfig(strategy="shuffled", seed=77)

    first = scheduler.build_plan(config)
    second = scheduler.build_plan(config)

    assert first.lesson_ids == second.lesson_ids
    assert first.lesson_ids != sorted(first.lesson_ids)


def test_balanced_strategy_returns_multiple_skills(tmp_path) -> None:
    scheduler = CurriculumScheduler(indexed_store=build_store(tmp_path))

    plan = scheduler.build_plan(CurriculumConfig(strategy="balanced"))

    assert set(plan.counts_by_skill) >= {"debugging", "repair"}


def test_module_targeted_strategy_filters_by_module(tmp_path) -> None:
    scheduler = CurriculumScheduler(indexed_store=build_store(tmp_path))

    plan = scheduler.build_plan(
        CurriculumConfig(strategy="module_targeted", target_modules=["debugging"])
    )

    assert plan.total > 0
    assert plan.counts_by_target_module["debugging"] == plan.total


def test_difficulty_filters_work(tmp_path) -> None:
    scheduler = CurriculumScheduler(indexed_store=build_store(tmp_path))

    plan = scheduler.build_plan(
        CurriculumConfig(strategy="sequential", difficulty_min=5, difficulty_max=5)
    )

    assert plan.total >= 0
    loaded = scheduler.load_lessons(plan)
    assert all(lesson.difficulty == 5 for lesson in loaded)


def test_verification_status_filters_work(tmp_path) -> None:
    scheduler = CurriculumScheduler(indexed_store=build_store(tmp_path))

    plan = scheduler.build_plan(
        CurriculumConfig(
            strategy="sequential",
            verification_statuses=["verified_target"],
        )
    )

    assert plan.total == 1
    assert plan.counts_by_verification_status == {"verified_target": 1}


def test_repair_boosted_places_repair_first(tmp_path) -> None:
    scheduler = CurriculumScheduler(indexed_store=build_store(tmp_path))

    plan = scheduler.build_plan(CurriculumConfig(strategy="repair_boosted"))
    lessons = scheduler.load_lessons(plan)

    assert lessons[0].skill == "repair"
    assert plan.metadata["duplicate_ids_used"] is False


def test_iter_batches_yields_expected_sizes(tmp_path) -> None:
    scheduler = CurriculumScheduler(indexed_store=build_store(tmp_path))
    plan = scheduler.build_plan(CurriculumConfig(strategy="sequential", batch_size=2))

    batches = list(scheduler.iter_batches(plan, batch_size=2))

    assert all(1 <= len(batch) <= 2 for batch in batches)
    assert sum(len(batch) for batch in batches) == plan.total


def test_load_lessons_works_with_indexed_store(tmp_path) -> None:
    scheduler = CurriculumScheduler(indexed_store=build_store(tmp_path))
    plan = scheduler.build_plan(CurriculumConfig(strategy="sequential", max_lessons=2))

    lessons = scheduler.load_lessons(plan)

    assert [lesson.id for lesson in lessons] == plan.lesson_ids


def test_empty_query_returns_empty_plan(tmp_path) -> None:
    scheduler = CurriculumScheduler(indexed_store=build_store(tmp_path))

    plan = scheduler.build_plan(CurriculumConfig(domains=["missing-domain"]))

    assert plan.total == 0
    assert plan.lesson_ids == []
    assert plan.counts_by_skill == {}


def test_plan_json_serialization(tmp_path) -> None:
    scheduler = CurriculumScheduler(indexed_store=build_store(tmp_path))
    plan = scheduler.build_plan(CurriculumConfig(strategy="balanced", max_lessons=3))

    path = plan.save_json(tmp_path / "plan.json")

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["total"] == plan.total
    assert loaded["lesson_ids"] == plan.lesson_ids
