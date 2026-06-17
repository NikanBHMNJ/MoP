# Colab 100M Training Notebook

Notebook:

```text
notebooks/train_100m_mopforge_colab.ipynb
```

This notebook is the first safe-by-default path for a real from-scratch
MoP-Forge GPU experiment on Google Colab L4, A100, or H100 runtimes. It is not
a production training guarantee and does not run large jobs automatically.

## Setup

Use a GPU runtime, then run the clone/install cell:

```bash
git clone https://github.com/NikanBHMNJ/MoP.git
cd MoP
pip install -e .[dev]
pip install datasets
```

The notebook also supports being opened from an already cloned repo.

## Environment Checks

The first cells check:

- `nvidia-smi`
- Python version
- PyTorch version
- CUDA availability
- `mopforge version`
- `mopforge doctor`
- `mopforge runtime detect`

If CUDA is unavailable, tiny smoke may still use CPU fallback, but that does
not validate GPU performance.

## Dataset Workflow

Default dataset:

```text
roneneldan/TinyStories
```

The notebook installs Hugging Face `datasets`, streams records where possible,
limits records with `MAX_RECORDS`, and writes:

```text
data/colab_tinystories_corpus.jsonl
```

Records are converted to MoP-Forge `TextCorpusRecord` JSONL using:

```bash
python scripts/build_colab_hf_corpus.py \
  --dataset roneneldan/TinyStories \
  --split train \
  --text-field text \
  --max-records 2000 \
  --output data/colab_tinystories_corpus.jsonl \
  --streaming
```

An optional disabled cell shows how to sample
`HuggingFaceH4/CodeAlpaca_20K` into simple instruction/output text records.

## Tiny Smoke First

Run:

```bash
mopforge gpu validate configs/jobs/tiny_gpu_smoke.json
mopforge gpu estimate configs/jobs/tiny_gpu_smoke.json
mopforge gpu train configs/jobs/tiny_gpu_smoke.json
```

The notebook generates `data/coding_bugfix_lessons.jsonl` if the tiny profile
needs it.

## 100M Dense/MoP

The notebook validates and estimates:

```bash
mopforge gpu validate configs/jobs/100m_dense_a100_smoke.json
mopforge gpu estimate configs/jobs/100m_dense_a100_smoke.json
mopforge gpu validate configs/jobs/100m_mop_a100_smoke.json
mopforge gpu estimate configs/jobs/100m_mop_a100_smoke.json
```

Training is opt-in:

```python
RUN_100M_DENSE = False
RUN_100M_MOP = False
```

Short Colab variants are written under `outputs/colab_configs/` with small
`max_steps`, `micro_batch_size=1`, configurable gradient accumulation,
`precision=auto`, and activation-checkpoint metadata enabled.

L4 24GB can be tight depending on sequence length, optimizer state, and model
shape. A100/H100 are the intended first serious hardware targets.

## Benchmark And Comparison

After runs finish, use:

```bash
mopforge gpu benchmark <run_id>
mopforge gpu show <run_id>
```

The notebook comparison cell loads local `metrics.json` and `state.json` files
and prints final train/eval loss, tokens seen, tokens/sec if present, peak
reserved VRAM if recorded, trainable parameter ratio, active parameter estimate,
routing density, and checkpoint path.

## Google Drive Backup

Set:

```python
BACKUP_TO_DRIVE = True
```

Then the notebook mounts Drive and copies:

```text
gpu_runs/
artifacts/
benchmarks/
reports/
outputs/
data/colab_tinystories_corpus.jsonl
```

to:

```text
/content/drive/MyDrive/mopforge_colab_runs/
```

## Limitations

- No required Hugging Face login for the default public dataset.
- No large model training by default.
- 100M jobs are opt-in and should follow tiny smoke.
- Colab session resets can delete local files before Drive backup.
- The memory estimator is approximate.
- This is not a guarantee that 100M or larger jobs fit every L4/A100/H100
  runtime.
- MoP routing remains PyTorch-level experimental plumbing, not fused kernels.
