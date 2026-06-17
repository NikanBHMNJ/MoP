from mopforge.gpu import build_module_routing_plan, group_batch_by_modules, routing_density


def test_routing_plan_density_and_grouping():
    known = ["core", "coding", "debugging"]
    active = [["coding"], ["debugging"], ["coding", "debugging"]]
    plan = build_module_routing_plan(active, known)
    assert plan.batch_size == 3
    assert plan.active_by_module["core"] == [0, 1, 2]
    assert routing_density(plan.active_by_sample, known) == plan.density
    groups = group_batch_by_modules(active, known)
    assert ("core", "coding") in groups
