# Installation

CPU/dev install:

```bash
pip install -e .[dev]
```

Minimal editable install:

```bash
pip install -e .
```

Optional PyTorch install:

```bash
pip install -e .[torch]
```

Optional tokenizer/Hugging Face compatibility:

```bash
pip install -e .[hf]
```

For CUDA, install the PyTorch build that matches your driver, CUDA runtime, and
platform using the official PyTorch instructions. MoP-Forge does not hardcode a
CUDA wheel URL and does not require CUDA to import or run CPU tests.

Windows notes: use PowerShell, keep paths quoted when they contain spaces, and
run `mopforge doctor` after installing optional dependencies.
