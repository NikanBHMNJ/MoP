"""Script entrypoint for comparing MoP-Forge GPU run efficiency metrics."""

from __future__ import annotations

from mopforge.gpu.compare import (
    COMPARISON_FIELDS,
    compare_runs,
    extract_run_row,
    format_table,
    load_run_payloads,
    main,
    resolve_run_dir,
    write_csv,
    write_json,
)

__all__ = [
    "COMPARISON_FIELDS",
    "compare_runs",
    "extract_run_row",
    "format_table",
    "load_run_payloads",
    "main",
    "resolve_run_dir",
    "write_csv",
    "write_json",
]


if __name__ == "__main__":
    raise SystemExit(main())
