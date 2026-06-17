"""Build a conservative paper-style Markdown scaffold."""

from __future__ import annotations

from mopforge.papers import PaperReportConfig, build_paper_report


def main() -> None:
    print("Paper report scaffold only. No PDF and no quality claims.")
    record = build_paper_report(
        PaperReportConfig(
            title="MoP-Forge CPU Smoke Report",
            abstract="Conservative local report scaffold for CPU-smoke artifacts.",
            dataset_refs=["coding_bugfix"],
            model_refs=["tiny_mop_oracle"],
        )
    )
    print(f"paper_report_id={record.paper_report_id}")
    print(f"status={record.status}")
    print(f"report_path={record.report_path}")


if __name__ == "__main__":
    main()
