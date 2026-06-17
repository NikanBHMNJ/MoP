# Architecture

MoP-Forge is a local filesystem research stack. Lessons, datasets, configs,
run records, checkpoints, and reports are plain files so experiments remain
inspectable and reproducible.

```text
KnowledgeLesson / Corpus
    -> Dataset Registry
    -> Tokenizer
    -> Trainer / GPUTrainer
    -> Model Registry / MoP Model
    -> Checkpoints / Artifacts
    -> Benchmark
    -> Analysis
    -> Paper Report
```

MoP parameter-family view:

```text
Stable Core
+ Module Parameters
+ Router Parameters
+ Fast Adapters
+ Generated Parameters
+ Feedback/Curriculum Control
```

Lessons are not neural weights. They are structured training signals and
provenance records. Dataset and SQLite stores keep examples, metadata, splits,
feedback, and curriculum state. Actual learned weights live in models,
checkpoints, and artifact directories.

Fast parameters currently mean two experimental local paths: named fast
adapters and generated adapter tensors. They are not custom CUDA kernels and
are not yet proven at large scale.
