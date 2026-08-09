from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ExecutionMetrics:
    planning_ms: float = 0.0
    execution_ms: float = 0.0
    object_checks: int = 0
    object_denials: int = 0
    policy_evaluations: int = 0
    policy_denials: int = 0
    property_checks: int = 0
    node_scan_candidates: int = 0
    edge_candidates: int = 0
    denied_before_enqueue: int = 0
    nodes_enqueued: int = 0
    edges_enqueued: int = 0
    bindings_materialized: int = 0
    update_candidates: int = 0
    with_check_evaluations: int = 0

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)
