"""Write simple statistical tables from local rows."""

from __future__ import annotations

from mopforge.statistics import make_metric_table, write_table_csv, write_table_json, write_table_markdown


def main() -> None:
    rows = [
        {"mode": "adapter", "final_eval_loss": 0.4},
        {"mode": "adapter", "final_eval_loss": 0.5},
        {"mode": "generated", "final_eval_loss": 0.3},
        {"mode": "generated", "final_eval_loss": 0.35},
    ]
    table = make_metric_table(rows, "mode", ["final_eval_loss"])
    print("Simple statistical tables only. No significance claims.")
    print(f"rows={len(table)}")
    print(f"json_path={write_table_json(table, 'outputs/stats/statistical_tables.json')}")
    print(f"csv_path={write_table_csv(table, 'outputs/stats/statistical_tables.csv')}")
    print(f"markdown_path={write_table_markdown(table, 'outputs/stats/statistical_tables.md')}")


if __name__ == "__main__":
    main()
