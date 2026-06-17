"""Statistical summaries and table helpers."""

from mopforge.statistics.summaries import (
    compare_groups_simple,
    mean,
    median,
    percent_change,
    stderr,
    stddev,
    summarize_by_group,
    summarize_metric,
)
from mopforge.statistics.tables import (
    make_metric_table,
    write_table_csv,
    write_table_json,
    write_table_markdown,
)

__all__ = [
    "compare_groups_simple",
    "make_metric_table",
    "mean",
    "median",
    "percent_change",
    "stderr",
    "stddev",
    "summarize_by_group",
    "summarize_metric",
    "write_table_csv",
    "write_table_json",
    "write_table_markdown",
]
