# Security

MoP-Forge is a local research framework, not a sandbox.

The local Python verifier (`mopforge.verify.verify_python_solution`) writes
candidate code and tests to a temporary file and runs them with the local Python
interpreter. Only run trusted code. A real sandbox such as Docker, a VM, or a
restricted execution service is outside the current implementation.

Do not put secrets in configs, manifests, dataset metadata, run records,
artifacts, or report files. The project currently stores local records as JSON,
JSONL, SQLite, and PyTorch checkpoint files without secret redaction.
