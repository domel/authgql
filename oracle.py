"""Independent, deliberately small oracle: no imports from authgql.

It consumes dictionaries and enumerates bounded walks by Cartesian products,
not executor adjacency expansion. Walk policies use equality over `flag`;
separate routines cover generated old/new-state SET checks, owner/dependency
snapshots and strict READ diagnostics. These are bounded generated sublanguages,
not a complete query parser or a general authorization implementation.
"""

from collections import Counter
from itertools import product


def identities(catalog, user):
    roles = set(catalog["user_roles"].get(user, []))
    while True:
        grown = roles | {parent for r in roles for parent in catalog["roles"][r]}
        if grown == roles:
            return {("USER", user)} | {("ROLE", r) for r in roles}
        roles = grown


def allowed(catalog, user, element, kind):
    who = identities(catalog, user)
    # Generator guarantees a broad permit, with optional node-label denials.
    for f in catalog["privileges"]:
        if (f.get("grantee_kind", "USER"), f["grantee"]) not in who:
            continue
        if f.get("effect") != "DENY" or not set(f["actions"]).intersection({"TRAVERSE", "MATCH"}):
            continue
        target = f["target"]
        if target.get("kind", "GRAPH") == "GRAPH":
            return False
        labels = element.get("labels", [element.get("type")])
        classes = target.get("labels" if kind == "NODE" else "edge_types", [])
        if target.get("kind") == ("NODES" if kind == "NODE" else "EDGES"):
            if not classes or set(classes).intersection(labels):
                return False
    scope = []
    for p in catalog["policies"]:
        if not p.get("enabled", True) or "TRAVERSE" not in p["actions"]:
            continue
        selector = p["selector"]
        if selector["kind"] != kind:
            continue
        if kind == "NODE" and not set(selector["labels"]).intersection(element["labels"]):
            continue
        if kind == "EDGE" and not set(element.get("labels", [element.get("type")])).intersection(
            selector["edge_types"]
        ):
            continue
        scope.append(p)
    if not scope:
        return True
    relevant = [
        p
        for p in scope
        if any(
            (identity["kind"], identity["name"]) in who
            for identity in p.get(
                "principals",
                [{"kind": p["grantee_kinds"][name], "name": name} for name in p["grantees"]],
            )
        )
    ]
    values = []
    for p in relevant:
        spec = p["using"]
        if isinstance(spec, bool):
            truth = spec
        else:
            observed = element.get("properties", {}).get("flag")
            truth = None if observed is None else observed == spec["property_compare"]["value"]
        values.append((p["effect"], truth))
    if any(effect == "DENY" and truth is not False for effect, truth in values):
        return False
    return any(effect == "PERMIT" and truth is True for effect, truth in values)


def walks(graph, catalog, user, bound, *, minimum=1, direction="out", paths=False, labels=()):
    nodes = {n["id"] for n in graph["nodes"] if allowed(catalog, user, n, "NODE")}
    edges = []
    for edge in graph["edges"]:
        if (
            edge["source"] not in nodes
            or edge["target"] not in nodes
            or not allowed(catalog, user, edge, "EDGE")
            or not set(labels).issubset(edge.get("labels", [edge.get("type")]))
        ):
            continue
        directed = edge.get("directed", True)
        if direction == "out" and directed:
            pairs = [(edge["source"], edge["target"])]
        elif direction == "in" and directed:
            pairs = [(edge["target"], edge["source"])]
        elif direction == "both" or (direction == "undirected" and not directed):
            pairs = list(
                dict.fromkeys([(edge["source"], edge["target"]), (edge["target"], edge["source"])])
            )
        else:
            pairs = []
        edges.extend((edge["id"], source, target) for source, target in pairs)
    answers = Counter()
    if minimum == 0:
        for node in nodes:
            answers[((("node", node),) if paths else (node, node))] += 1
    for length in range(max(1, minimum), bound + 1):
        for sequence in product(edges, repeat=length):
            if all(a[2] == b[1] for a, b in zip(sequence, sequence[1:])):
                key = (sequence[0][1], sequence[-1][2])
                if paths:
                    key = (("node", sequence[0][1]),) + tuple(
                        item
                        for edge, _, target in sequence
                        for item in [("edge", edge), ("node", target)]
                    )
                answers[key] += 1
    return answers


def owner_references_valid(policies, owners, root):
    """Independent acyclic dependency oracle; owners maps to grant/deny flags."""
    pending, seen = [root], set()
    while pending:
        name = pending.pop()
        if name not in policies:
            return False
        if name in seen:
            continue
        seen.add(name)
        policy = policies[name]
        grant, deny = owners.get(policy["owner"], (False, False))
        if not grant or deny:
            return False
        pending.extend(policy["depends"])
    return True


def flag_update(graph, selected, value, ceiling, reference_valid=True):
    """Oracle for the generated old/new-state SET policy, not an SQL parser.

    Old flag >= 0; new flag <= ceiling; no differently flagged N node in
    the complete prospective graph.
    """
    import copy

    before = copy.deepcopy(graph)
    proposed = copy.deepcopy(graph)
    chosen = [n for n in proposed["nodes"] if n["id"] in selected]
    if not chosen:
        return None, proposed

    def number(value):
        return type(value) in (int, float)

    if not reference_valid:
        return "42000", before
    if any(not number(n["properties"].get("flag")) or n["properties"]["flag"] < 0 for n in chosen):
        return "42000", before
    for node in chosen:
        node["properties"]["flag"] = value
    if not number(value) or value > ceiling:
        return "42000", before
    for node in proposed["nodes"]:
        flag = node["properties"].get("flag")
        if flag is not None and (not number(flag) or flag != value):
            return "42000", before
    return None, proposed


def read_outcome(flag, selected, sorted_key):
    """Generated one-node query: READ policy requires numeric flag = 1."""
    if not selected:
        return None, []
    if type(flag) not in (int, float) or flag != 1:
        return "42000", None
    return None, [{"value": flag}]
