"""Reproducible interpreter microbenchmarks; NOT a native GQL DBMS benchmark.

Run: python benchmark.py --output results/benchmark.json
21 measured repetitions, two warmups, one separately traced allocation run.
Generation/parsing excluded; executor construction/compilation included.
"""

from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time
import tracemalloc

from authgql.authorization import AuthorizationEngine
from authgql.catalog import AuthorizationCatalog, PolicyDescriptor
from authgql.executor import SecureExecutor
from authgql.model import PropertyGraph, Node, Edge
from authgql.parser import parse_query


class DisabledAuthorization(AuthorizationEngine):
    """Benchmark-only bypass, never a deployable enforcement mode."""

    def check_access(self):
        pass

    def check(self, *args, **kwargs):
        pass

    def check_with(self, *args, **kwargs):
        pass


class ReadMemoization(AuthorizationEngine):
    """Benchmark adapter valid only for one immutable read statement.

    Graph/catalog/user/parameters fixed by construction. No cross-statement or
    prospective-state caching. Keys include the property key; denied decisions
    are not cached. This adapter is never used in the normal session executor.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = set()
        self.hits = 0

    def check(self, action, resource, property_name=None, graph=None):
        assert graph is None or graph is self.graph
        key = (action, resource.kind, resource.id, property_name)
        if key in self.cache:
            self.hits += 1
            return
        super().check(action, resource, property_name, graph)
        self.cache.add(key)


def ring(n, branching=1):
    nodes = {
        str(i): Node(str(i), {"N"} | ({"Start", "Target"} if i == 0 else set()), {"flag": 1})
        for i in range(n)
    }
    edges = {
        f"{i}:{j}": Edge(f"{i}:{j}", "L", str(i), str((i + j + 1) % n), {"flag": 1})
        for i in range(n)
        for j in range(branching)
    }
    return PropertyGraph("g", nodes, edges)


def catalog(count=0, depth=1, predicate="property", update=False):
    roles = {"r0": []} | {f"r{i}": [f"r{i - 1}"] for i in range(1, depth)}
    c = AuthorizationCatalog.from_dict(
        {
            "roles": roles,
            "user_roles": {"u": [f"r{depth - 1}"]},
            "privileges": [
                {"grantee": "r0", "actions": ["ACCESS", "MATCH", "SET"], "target": {"graph": "g"}},
                {
                    "grantee": "owner",
                    "grantee_kind": "USER",
                    "actions": ["POLICY REFERENCE"],
                    "target": {"graph": "g"},
                },
            ],
        }
    )
    predicates = {
        "property": "RESOURCE.flag = 1",
        "exists1": "EXISTS { MATCH (RESOURCE)-[:L]->(n) }",
        "exists3": "EXISTS { MATCH (RESOURCE)-[:L]->{1,3}(n) }",
    }
    for i in range(count):
        c.add_policy(
            PolicyDescriptor(
                f"p{i}",
                "g",
                {"TRAVERSE"},
                {"r0"},
                "PERMIT",
                owner="owner",
                selector={"kind": "NODE", "labels": ["N"]},
                using=predicates[predicate],
                grantee_kinds={"r0": "ROLE"},
            )
        )
    if update:
        c.add_policy(
            PolicyDescriptor(
                "post",
                "g",
                {"SET"},
                {"r0"},
                "PERMIT",
                owner="owner",
                with_check="NEW_RESOURCE.flag = 1",
                grantee_kinds={"r0": "ROLE"},
            )
        )
    return c


def measure(callback, repeats):
    for _ in range(2):
        callback()
    times = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter_ns()
        callback()
        times.append((time.perf_counter_ns() - start) / 1e6)
    gc.collect()
    tracemalloc.start()
    metrics = callback()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "samples_ms": times,
        "p50_ms": statistics.median(times),
        "p95_ms": sorted(times)[math.ceil(0.95 * len(times)) - 1],
        "peak_extra_mib": peak / (1024**2),
        "metrics": metrics,
    }


def run_case(spec, repeats):
    g = ring(spec["nodes"], spec.get("branch", 1))
    c = catalog(
        spec.get("policies", 0),
        spec.get("depth", 1),
        spec.get("predicate", "property"),
        spec.get("kind") == "update",
    )
    bound = spec.get("hops", 1)
    if spec.get("kind") == "update":
        source = (
            f"MATCH (n:{'N' if spec.get('delta') == 'all' else 'Target'}) SET n.flag = 1 FINISH"
        )
    elif spec.get("kind") == "path":
        source = f"MATCH (s:Start)-[:L]->{{1,{bound}}}(n:N) RETURN COUNT(n) AS n"
    else:
        source = "MATCH (n:N) RETURN COUNT(n) AS n"
    q = parse_query(source)

    def callback():
        if spec.get("kind") == "clone":
            cloned = g.clone()
            return {"cloned_nodes": len(cloned.nodes), "cloned_edges": len(cloned.edges)}
        ex = SecureExecutor(g, c, "u", preflight=spec.get("mode") != "disabled")
        if spec.get("mode") == "disabled":
            ex.authorization = DisabledAuthorization(c, g, "u", metrics=ex.metrics)
        if spec.get("mode") == "memo":
            ex.authorization = ReadMemoization(c, g, "u", metrics=ex.metrics)
        result = ex.execute(q)
        metrics = result.metrics.to_dict()
        metrics["cache_hits"] = getattr(ex.authorization, "hits", 0)
        if q.update_kind:
            expected = spec["nodes"] if spec.get("delta") == "all" else 1
            assert result.affected_elements == expected, (spec, result.affected_elements)
            assert len(g.nodes) == spec["nodes"] and len(g.edges) == spec["nodes"]
        else:
            if spec.get("kind") == "path":
                expected = sum(spec.get("branch", 1) ** h for h in range(1, bound + 1))
            else:
                expected = spec["nodes"]
            assert result.rows == [{"n": expected}], (spec, result.rows, expected)
        return metrics

    measured = measure(callback, repeats)
    if q.update_kind:
        assert all(node.properties["flag"] == 1 for node in g.nodes.values())
    return {**spec, "edges": len(g.edges), "query": source, **measured}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="results/benchmark.json")
    ap.add_argument("--repeats", type=int, default=21)
    args = ap.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    cases = []
    for n in [1000, 10000, 100000]:
        for mode, count in [("disabled", 0), ("object", 0), ("policy", 1)]:
            cases.append({"family": "size", "nodes": n, "mode": mode, "policies": count})
    for count in [1, 10, 100]:
        cases.append({"family": "policies", "nodes": 1000, "policies": count})
    for depth in [1, 2, 4, 8, 16]:
        cases.append({"family": "roles", "nodes": 1000, "policies": 1, "depth": depth})
    for pred in ["property", "exists1", "exists3"]:
        cases.append({"family": "predicate", "nodes": 1000, "policies": 1, "predicate": pred})
    for hops in [1, 3, 6, 12]:
        cases.append({"family": "path", "kind": "path", "nodes": 1000, "policies": 1, "hops": hops})
    for branch in [2, 4, 8, 16]:
        for mode in ["policy", "memo"]:
            cases.append(
                {
                    "family": "branch",
                    "kind": "path",
                    "nodes": 1000,
                    "policies": 1,
                    "branch": branch,
                    "hops": 3,
                    "mode": mode,
                }
            )
    for n in [1000, 10000, 100000]:
        cases.append({"family": "writes", "kind": "clone", "nodes": n})
        # Current kind-qualified lookup supports the same size range for bulk
        # updates. Full graph staging/publication is still included in timing.
        for delta in ["one", "all"]:
            cases.append({"family": "writes", "kind": "update", "nodes": n, "delta": delta})
    cpu = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        platform.processor(),
    )
    report = {
        "metadata": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu": cpu,
            "logical_cpus": os.cpu_count(),
            "ram_gib": os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024**3),
            "repetitions": args.repeats,
            "warmups": 2,
            "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "memory_metric": "additional tracemalloc peak; input graph allocated before tracing",
            "latency_metric": "wall time without tracemalloc; GC collected before each sample",
            "source_sha256": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [
                    *sorted(Path("authgql").glob("*.py")),
                    Path(__file__).relative_to(Path.cwd()),
                ]
            },
        },
        "cases": [],
    }
    for index, spec in enumerate(cases):
        result = run_case(spec, args.repeats)
        report["cases"].append(result)
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"{index + 1}/{len(cases)} {spec}: p50={result['p50_ms']:.3f} ms, "
            f"extra peak={result['peak_extra_mib']:.3f} MiB",
            flush=True,
        )
    for name, digest in report["metadata"]["source_sha256"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    report["metadata"]["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
