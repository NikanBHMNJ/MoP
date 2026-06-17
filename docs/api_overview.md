# Public API Overview

MoP-Forge v0.45.0 keeps broad top-level imports for backwards compatibility,
but new integrations should prefer `mopforge.public_api`.

## Stable API

Stable symbols cover local lessons, config envelopes, runtime planning, and the
CPU tiny trainer:

- `KnowledgeLesson`, `LessonStore`, `IndexedLessonStore`, `LessonDataset`
- `MoPForgeConfig`, `get_default_config`, `list_default_config_names`
- `RuntimeConfig`, `RuntimeContext`, `DeviceInfo`, `PrecisionPolicy`
- `build_runtime_context`, `detect_devices`
- `TrainerConfig`, `TrainerState`, `TrainerResult`, `TinyTrainer`

## Experimental API

Experimental symbols are usable but may change before v1.0:

- `GPUTrainingConfig`, `GPUTrainingState`, `GPUTrainingResult`
- `GPUDataConfig`, `GPUTrainer`, `AmpScaler`
- `DistributedConfig`, `ModelMemoryEstimate`, `estimate_training_memory`

## Internal Modules

Underscore-prefixed helpers and module-private functions are internal. The
top-level `mopforge` namespace remains broad for older examples, but it should
not be treated as a promise that every helper is stable.

## Compatibility Policy

Stable imports should remain import-compatible across minor release-candidate
updates. Experimental APIs can gain fields or change behavior while MoP-Forge
moves toward v1.0-beta. Breaking changes should be documented in release notes.
