"""Demonstrate full local checkpoint resume for TinyTrainer, SFT, and CPT."""

from __future__ import annotations

from pathlib import Path

from mopforge.artifacts import ArtifactManager, CheckpointManager
from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.kts import IndexedLessonStore, LessonStore
from mopforge.pretrain import (
    ContinuedPretrainConfig,
    TextCorpusStore,
    build_demo_code_corpus,
    run_continued_pretraining,
)
from mopforge.sft import FinetuneConfig, run_finetune
from mopforge.training import TinyTrainer, TrainerConfig


ROOT = Path(__file__).resolve().parents[1]
BUGFIX_PATH = ROOT / "data" / "coding_bugfix_lessons.jsonl"
LESSON_PATH = ROOT / "data" / "indexed_lessons.jsonl"
INDEX_PATH = ROOT / "data" / "kts_index.sqlite"
CORPUS_PATH = ROOT / "data" / "text_corpus.jsonl"
RUNS_ROOT = ROOT / "runs"
ARTIFACT_ROOT = ROOT / "artifacts"


def main() -> None:
    """Run tiny initial/resumed lifecycle smoke jobs."""

    print(
        "CPU smoke checkpoint-resume MVP only. Metrics are not model-quality claims."
    )
    ensure_bugfix_lessons()
    ensure_indexed_kts()
    ensure_corpus()

    trainer_original = TinyTrainer(_trainer_config(max_steps=1)).train()
    trainer_checkpoint = latest_full_checkpoint(trainer_original.run_id, "trainer")
    trainer_resumed = TinyTrainer(
        _trainer_config(
            max_steps=2,
            resume_from_checkpoint=trainer_checkpoint.path,
        )
    ).train()
    print_resume_block(
        "trainer",
        trainer_original.run_id,
        trainer_checkpoint.artifact_id,
        trainer_checkpoint.path,
        trainer_original.metrics["global_step"],
        trainer_resumed.run_id,
        trainer_resumed.metrics["global_step"],
        trainer_resumed.artifacts["trainer_result_json"],
    )

    sft_original = run_finetune(_sft_config(max_steps=1))
    sft_checkpoint = latest_full_checkpoint(sft_original.run_id, "sft")
    sft_resumed = run_finetune(
        _sft_config(max_steps=2, resume_from_checkpoint=sft_checkpoint.path)
    )
    print_resume_block(
        "sft",
        sft_original.run_id,
        sft_checkpoint.artifact_id,
        sft_checkpoint.path,
        sft_original.metrics["global_step"],
        sft_resumed.run_id,
        sft_resumed.metrics["global_step"],
        sft_resumed.artifacts["finetune_result_json"],
    )

    cpt_original = run_continued_pretraining(_cpt_config(max_steps=1))
    cpt_checkpoint = latest_full_checkpoint(cpt_original.run_id, "pretrain")
    cpt_resumed = run_continued_pretraining(
        _cpt_config(max_steps=2, resume_from_checkpoint=cpt_checkpoint.path)
    )
    print_resume_block(
        "pretrain",
        cpt_original.run_id,
        cpt_checkpoint.artifact_id,
        cpt_checkpoint.path,
        cpt_original.metrics["global_step"],
        cpt_resumed.run_id,
        cpt_resumed.metrics["global_step"],
        cpt_resumed.artifacts["continued_pretrain_result_json"],
    )


def _trainer_config(*, max_steps: int, resume_from_checkpoint: str | None = None) -> TrainerConfig:
    return TrainerConfig(
        run_name="resume_demo_trainer",
        model_type="mop_oracle",
        lesson_path=str(LESSON_PATH),
        index_path=str(INDEX_PATH),
        run_registry_root=str(RUNS_ROOT),
        artifact_root=str(ARTIFACT_ROOT),
        max_steps=max_steps,
        eval_interval=1,
        eval_batches=1,
        batch_size=1,
        max_seq_len=128,
        d_model=16,
        n_layers=1,
        n_heads=2,
        save_checkpoints=False,
        save_full_checkpoints=True,
        resume_from_checkpoint=resume_from_checkpoint,
    )


def _sft_config(*, max_steps: int, resume_from_checkpoint: str | None = None) -> FinetuneConfig:
    return FinetuneConfig(
        mode="sft_full",
        lesson_path=str(LESSON_PATH),
        index_path=str(INDEX_PATH),
        run_registry_root=str(RUNS_ROOT),
        artifact_root=str(ARTIFACT_ROOT),
        max_steps=max_steps,
        eval_batches=1,
        batch_size=1,
        max_seq_len=128,
        save_checkpoints=False,
        save_full_checkpoints=True,
        resume_from_checkpoint=resume_from_checkpoint,
    )


def _cpt_config(
    *,
    max_steps: int,
    resume_from_checkpoint: str | None = None,
) -> ContinuedPretrainConfig:
    return ContinuedPretrainConfig(
        run_name="resume_demo_pretrain",
        corpus_path=str(CORPUS_PATH),
        lesson_path=str(LESSON_PATH),
        run_registry_root=str(RUNS_ROOT),
        artifact_root=str(ARTIFACT_ROOT),
        max_steps=max_steps,
        eval_batches=1,
        batch_size=1,
        max_seq_len=96,
        d_model=16,
        n_layers=1,
        n_heads=2,
        save_checkpoints=False,
        save_full_checkpoints=True,
        resume_from_checkpoint=resume_from_checkpoint,
    )


def latest_full_checkpoint(run_id: str, training_kind: str):
    checkpoint = CheckpointManager(
        ArtifactManager(ARTIFACT_ROOT)
    ).latest_full_checkpoint(run_id=run_id, training_kind=training_kind)
    if checkpoint is None:
        raise RuntimeError(f"No full checkpoint found for run_id={run_id}")
    return checkpoint


def print_resume_block(
    kind: str,
    original_run_id: str,
    checkpoint_artifact_id: str,
    checkpoint_path: str,
    original_step: int,
    resumed_run_id: str,
    final_step: int,
    result_path: str,
) -> None:
    print(f"{kind}:")
    print(f"  original_run_id={original_run_id}")
    print(f"  checkpoint_artifact_id={checkpoint_artifact_id}")
    print(f"  checkpoint_path={checkpoint_path}")
    print(f"  original_global_step={original_step}")
    print(f"  resumed_run_id={resumed_run_id}")
    print(f"  resumed_final_global_step={final_step}")
    print(f"  result_path={result_path}")


def ensure_bugfix_lessons() -> None:
    if BUGFIX_PATH.exists():
        return
    lessons = generate_coding_bugfix_lessons(count_per_category=10, verify=True)
    LessonStore(BUGFIX_PATH).add_many(lesson for lesson in lessons if lesson.is_verified)


def ensure_indexed_kts() -> None:
    if LESSON_PATH.exists() and INDEX_PATH.exists():
        return
    lessons = LessonStore(BUGFIX_PATH).load_all()
    store = IndexedLessonStore(LESSON_PATH, INDEX_PATH)
    for lesson in lessons:
        store.add(lesson)


def ensure_corpus() -> None:
    if CORPUS_PATH.exists():
        return
    TextCorpusStore(CORPUS_PATH).add_many(build_demo_code_corpus(count=6))


if __name__ == "__main__":
    main()
