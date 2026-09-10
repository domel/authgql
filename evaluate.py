#!/usr/bin/env python3
"""Run the deterministic security-validation workload used in the article.

Wall-clock fields are deliberately omitted from the report.  The prototype
records them for exploratory profiling, but the paper reports only deterministic
results and counters unless a controlled benchmark environment is supplied.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from authgql.catalog import AuthorizationCatalog
from authgql.errors import AuthGQLError
from authgql.executor import SecureExecutor
from authgql.model import PropertyGraph
from authgql.parser import parse_query


ROOT = Path(__file__).resolve().parent
GRAPH_PATH = ROOT / "examples" / "hospital.graph.json"
CATALOG_PATH = ROOT / "examples" / "hospital.catalog.json"
COUNTERS = (
    "object_checks",
    "policy_evaluations",
    "property_checks",
    "node_scan_candidates",
    "edge_candidates",
    "denied_before_enqueue",
    "nodes_enqueued",
    "edges_enqueued",
    "bindings_materialized",
    "update_candidates",
    "with_check_evaluations",
)


def fresh() -> tuple[PropertyGraph, AuthorizationCatalog]:
    return PropertyGraph.load(GRAPH_PATH), AuthorizationCatalog.load(CATALOG_PATH)


def stable_metrics(result: Any) -> dict[str, int]:
    return {name: int(getattr(result.metrics, name)) for name in COUNTERS}


def read_case(name: str, source: str, user: str = "alice") -> dict[str, Any]:
    graph, catalog = fresh()
    result = SecureExecutor(graph, catalog, user).execute(
        parse_query(source), compare_reference=True
    )
    return {
        "name": name,
        "status": "ok",
        "rows": result.rows,
        "row_count": len(result.rows),
        "reference_equal": result.reference_equal,
        "metrics": stable_metrics(result),
    }


def denied_case(name: str, source: str, user: str) -> dict[str, Any]:
    graph, catalog = fresh()
    try:
        SecureExecutor(graph, catalog, user).execute(parse_query(source))
    except AuthGQLError as exc:
        return {"name": name, "status": "denied", "code": exc.code}
    return {"name": name, "status": "unexpectedly_permitted"}


def rejected_write_case() -> dict[str, Any]:
    graph, catalog = fresh()
    before = graph.to_dict()
    try:
        SecureExecutor(graph, catalog, "alice").execute(
            parse_query("MATCH (p:Patient) WHERE p.age = 42 SET p.classification = 'sealed' FINISH")
        )
    except AuthGQLError as exc:
        return {
            "name": "with_check_rejection",
            "status": "denied",
            "code": exc.code,
            "graph_unchanged": graph.to_dict() == before,
        }
    return {"name": "with_check_rejection", "status": "unexpectedly_permitted"}


def main() -> int:
    cases = [
        read_case(
            "guarded_fixed_path",
            "MATCH (p:Patient)-[:TREATED_AT]->(h:Hospital) "
            "OPTIONAL MATCH (p)-[:HAS_DIAGNOSIS]->(d:Diagnosis) "
            "RETURN p.age AS age, d.code AS diagnosis",
        ),
        read_case(
            "hidden_exists",
            "MATCH (h:Hospital {code:'H2'}) "
            "WHERE EXISTS { MATCH (p:Patient)-[:TREATED_AT]->(h) } "
            "RETURN h.code AS hospital",
        ),
        read_case(
            "authorized_aggregate",
            "MATCH (p:Patient) RETURN COUNT(p) AS patients",
        ),
        read_case(
            "guarded_quantified_path",
            "MATCH (p:Patient)-[:HAS_DIAGNOSIS*1..3]->(d:Diagnosis) RETURN d.code AS diagnosis",
        ),
        denied_case(
            "protected_property",
            "MATCH (p:Patient) RETURN p.name AS name",
            "rita",
        ),
        rejected_write_case(),
    ]
    read_cases = [case for case in cases if "reference_equal" in case]
    report = {
        "scenario": "hospital",
        "authorization_identifier": "alice except where stated",
        "cases": cases,
        "summary": {
            "case_count": len(cases),
            "all_read_cases_equal_reference": all(
                case["reference_equal"] is True for case in read_cases
            ),
            "denied_before_enqueue": sum(
                case["metrics"]["denied_before_enqueue"] for case in read_cases
            ),
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if all(case["status"] in {"ok", "denied"} for case in cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
