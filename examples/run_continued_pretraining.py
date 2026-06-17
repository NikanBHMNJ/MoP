"""Run a tiny continued-pretraining corpus API smoke example."""

from __future__ import annotations

from pathlib import Path

from mopforge.pretrain import (
    ContinuedPretrainConfig,
    TextCorpusStore,
    build_demo_code_corpus,
    run_continued_pretraining,
)


ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "data" / "text_corpus.jsonl"
RUNS_ROOT = ROOT / "runs"
ARTIFACT_ROOT = ROOT / "artifacts"


def main() -> None:
    """Build a tiny corpus and run continued-pretraining smoke training."""

    print(
        "CPU smoke continued-pretraining API only. Metrics are not "
        "model-quality claims."
    )
    build_demo_corpus()
    result = run_continued_pretraining(
        ContinuedPretrainConfig(
            run_name="tiny_continued_pretrain",
            corpus_path=str(CORPUS_PATH),
            lesson_path=None,
            run_registry_root=str(RUNS_ROOT),
            artifact_root=str(ARTIFACT_ROOT),
            max_steps=1,
            eval_batches=1,
            batch_size=2,
            max_seq_len=128,
            d_model=32,
            n_layers=1,
            n_heads=2,
        )
    )

    print(f"run_id={result.run_id}")
    print(f"corpus_records={result.corpus_records}")
    print(f"corpus_chunks={result.corpus_chunks}")
    print(
        f"train_loss={result.final_train_loss:.4f} "
        f"eval_loss={result.final_eval_loss:.4f}"
    )
    print(f"checkpoint_artifact_ids={result.metrics['checkpoint_artifact_ids']}")
    print(f"result_path={result.artifacts['continued_pretrain_result_json']}")


def build_demo_corpus() -> None:
    """Rebuild the deterministic demo corpus."""

    if CORPUS_PATH.exists():
        CORPUS_PATH.unlink()
    records = build_demo_code_corpus(count=12)
    TextCorpusStore(CORPUS_PATH).add_many(records)


if __name__ == "__main__":
    main()
