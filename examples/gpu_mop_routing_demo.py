"""Inspect MoP routing and fast-parameter metadata helpers."""

from __future__ import annotations

from mopforge.gpu import build_module_routing_plan, group_batch_by_modules


def main() -> None:
    known = ["core", "coding", "debugging", "repair"]
    active = [["coding", "debugging"], ["repair"], ["coding"]]
    plan = build_module_routing_plan(active, known)
    print(f"batch_size={plan.batch_size}")
    print(f"density={plan.density:.3f}")
    print(f"active_by_module={plan.active_by_module}")
    print(f"groups={group_batch_by_modules(active, known)}")


if __name__ == "__main__":
    main()
