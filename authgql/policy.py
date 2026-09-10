from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .errors import CatalogError
from .model import Edge, GraphElement, Node, PropertyGraph, PathValue
from .values import compare as typed_compare, parameter
from .validation import expression_type, require
from .parser import (
    BinaryExpr,
    ContextValue,
    ExistsExpr,
    Expr,
    FunctionCall,
    LabelTest,
    Literal,
    MatchClause,
    PatternChain,
    PatternNode,
    PropertyRef,
    UnaryExpr,
    VariableRef,
    parse_expression,
    scan_clauses,
    walk_expr,
)


@dataclass
class PolicyEvaluationContext:
    user: str
    roles: set[str]
    parameters: dict[str, Any] = field(default_factory=dict)
    resource: GraphElement | None = None
    new_resource: GraphElement | None = None
    remaining_steps: int = 100_000

    def tick(self) -> None:
        self.remaining_steps -= 1
        if self.remaining_steps < 0:
            raise CatalogError("Policy evaluation budget exhausted")


TruthValue = bool | None
Predicate = Callable[[PropertyGraph, PolicyEvaluationContext], TruthValue]


def _as_truth(value: Any) -> TruthValue:
    if value is True or value is False or value is None:
        return value
    raise CatalogError(f"Policy Boolean expression produced a non-Boolean value: {value!r}")


def _truth_not(value: Any) -> TruthValue:
    truth = _as_truth(value)
    return None if truth is None else not truth


def _truth_and(left: Any, right: Any) -> TruthValue:
    left_truth = _as_truth(left)
    right_truth = _as_truth(right)
    if left_truth is False or right_truth is False:
        return False
    if left_truth is None or right_truth is None:
        return None
    return True


def _truth_or(left: Any, right: Any) -> TruthValue:
    left_truth = _as_truth(left)
    right_truth = _as_truth(right)
    if left_truth is True or right_truth is True:
        return True
    if left_truth is None or right_truth is None:
        return None
    return False


def _compare(left: Any, op: str, right: Any) -> TruthValue:
    return typed_compare(left, op, right)


def _source(context: PolicyEvaluationContext, name: str) -> Any:
    if name.upper().startswith("PARAM:"):
        return parameter(context.parameters, name.split(":", 1)[1])
    name = name.upper()
    if name == "RESOURCE":
        return context.resource
    if name == "NEW_RESOURCE":
        return context.new_resource
    if name == "SESSION_USER":
        return context.user
    if name.startswith("PARAM:"):
        return parameter(context.parameters, name.split(":", 1)[1])
    raise CatalogError(f"Unknown policy source: {name}")


def _node_matches(node: Node, spec: dict[str, Any], context: PolicyEvaluationContext) -> bool:
    context.tick()
    labels = set(map(str, spec.get("labels", [])))
    if "label" in spec:
        labels.add(str(spec["label"]))
    if labels and not labels.issubset(node.labels):
        return False
    props = dict(spec.get("properties", {}))
    if "property" in spec:
        props[str(spec["property"])] = spec.get("value")
    for key, expected in props.items():
        if expected == "SESSION_USER":
            expected = context.user
        elif isinstance(expected, str) and expected.startswith("PARAM:"):
            expected = parameter(context.parameters, expected.split(":", 1)[1])
        if _compare(node.properties.get(key), "=", expected) is not True:
            return False
    return True


def _path_exists(
    graph: PropertyGraph, spec: dict[str, Any], context: PolicyEvaluationContext
) -> bool:
    start_spec = spec.get("start", {})
    starts: list[Node] = []
    if isinstance(start_spec, str):
        candidate = _source(context, start_spec)
        if isinstance(candidate, Node):
            starts = [candidate]
    elif "source" in start_spec:
        candidate = _source(context, str(start_spec["source"]))
        if isinstance(candidate, Node):
            starts = [candidate]
    else:
        starts = [node for node in graph.nodes.values() if _node_matches(node, start_spec, context)]

    steps = list(spec.get("steps", []))
    current = starts
    for step in steps:
        next_nodes: dict[str, Node] = {}
        edge_type = step.get("type")
        direction = str(step.get("direction", "out")).lower()
        edge_props = step.get("properties", {})
        for node in current:
            for edge, neighbour in graph.adjacent(node.id, direction):
                context.tick()
                if edge_type and edge_type not in edge.labels:
                    continue
                if any(
                    _compare(edge.properties.get(key), "=", value) is not True
                    for key, value in edge_props.items()
                ):
                    continue
                next_nodes[neighbour.id] = neighbour
        current = list(next_nodes.values())
        if not current:
            return False

    end_spec = spec.get("end", "RESOURCE")
    if isinstance(end_spec, str):
        candidate = _source(context, end_spec)
        return isinstance(candidate, Node) and any(node.id == candidate.id for node in current)
    if isinstance(end_spec, dict):
        if "source" in end_spec:
            candidate = _source(context, str(end_spec["source"]))
            return isinstance(candidate, Node) and any(node.id == candidate.id for node in current)
        return any(_node_matches(node, end_spec, context) for node in current)
    return False


def _resolve_context_value(value: Any, context: PolicyEvaluationContext) -> Any:
    if not isinstance(value, ContextValue):
        return value
    if value.name == "SESSION_USER":
        return context.user
    if value.name.startswith("$"):
        return parameter(context.parameters, value.name[1:])
    return None


def _gql_pattern_node_matches(
    node: Node,
    pattern: PatternNode,
    context: PolicyEvaluationContext,
) -> bool:
    context.tick()
    if pattern.labels and not pattern.labels.issubset(node.labels):
        return False
    return all(
        _compare(
            node.properties.get(name),
            "=",
            _resolve_context_value(expected, context),
        )
        is True
        for name, expected in pattern.properties.items()
    )


def _gql_expand_pattern(
    graph: PropertyGraph,
    start: Node,
    pattern: PatternChain,
    edge_index: int,
    row: dict[str, Any],
    context: PolicyEvaluationContext,
) -> list[tuple[dict[str, Any], PathValue]]:
    edge_pattern = pattern.edges[edge_index]
    node_pattern = pattern.nodes[edge_index + 1]
    output = []
    bound_node = row.get(node_pattern.variable)
    node_is_bound = node_pattern.variable in row
    edge_is_bound = bool(edge_pattern.variable and edge_pattern.variable in row)
    bound_edge = row.get(edge_pattern.variable) if edge_pattern.variable else None
    if edge_pattern.min_hops == 0:
        candidate_row = dict(row)
        node_ok = not node_is_bound or (isinstance(bound_node, Node) and bound_node.id == start.id)
        edge_ok = (
            not edge_pattern.variable
            or edge_pattern.variable not in row
            or row[edge_pattern.variable] == []
        )
        if (
            node_ok
            and edge_ok
            and _gql_pattern_node_matches(
                bound_node if isinstance(bound_node, Node) else start, node_pattern, context
            )
        ):
            if not node_is_bound:
                candidate_row[node_pattern.variable] = start
            if edge_pattern.variable and not edge_is_bound:
                candidate_row[edge_pattern.variable] = []
            output.append((candidate_row, PathValue((start.id,), ())))

    frontier: list[tuple[Node, list[Edge], list[str]]] = [(start, [], [start.id])]
    while frontier:
        current, path, path_nodes = frontier.pop()
        if len(path) >= edge_pattern.max_hops:
            continue
        for edge, neighbour in graph.adjacent(current.id, edge_pattern.direction):
            context.tick()
            if not edge_pattern.required_labels.issubset(edge.labels):
                continue
            # A correlated binding retains its supplied version, including the
            # old RESOURCE in WITH CHECK; the phase graph supplies topology.
            edge_for_test = (
                bound_edge if isinstance(bound_edge, Edge) and bound_edge.id == edge.id else edge
            )
            if any(
                _compare(
                    edge_for_test.properties.get(name),
                    "=",
                    _resolve_context_value(expected, context),
                )
                is not True
                for name, expected in edge_pattern.properties.items()
            ):
                continue
            new_path = [*path, edge]
            node_ok = (not node_is_bound) or (
                isinstance(bound_node, Node) and bound_node.id == neighbour.id
            )
            if (
                len(new_path) >= edge_pattern.min_hops
                and node_ok
                and _gql_pattern_node_matches(
                    bound_node if isinstance(bound_node, Node) else neighbour, node_pattern, context
                )
            ):
                candidate_row = dict(row)
                edge_value: Edge | list[Edge] = (
                    new_path[0] if not edge_pattern.quantified else new_path
                )
                edge_ok = (
                    not edge_pattern.variable
                    or not edge_is_bound
                    or _compare(bound_edge, "=", edge_value) is True
                )
                if node_ok and edge_ok:
                    if not node_is_bound:
                        candidate_row[node_pattern.variable] = neighbour
                    if edge_pattern.variable and not edge_is_bound:
                        candidate_row[edge_pattern.variable] = edge_value
                    output.append(
                        (
                            candidate_row,
                            PathValue(
                                tuple([*path_nodes, neighbour.id]), tuple(e.id for e in new_path)
                            ),
                        )
                    )
            if len(new_path) < edge_pattern.max_hops:
                frontier.append((neighbour, new_path, [*path_nodes, neighbour.id]))
    return output


def _gql_match_clause(
    graph: PropertyGraph,
    clause: MatchClause,
    input_rows: list[dict[str, Any]],
    context: PolicyEvaluationContext,
) -> list[dict[str, Any]]:
    pattern = clause.pattern
    output: list[dict[str, Any]] = []
    for input_row in input_rows:
        first_pattern = pattern.nodes[0]
        if first_pattern.variable in input_row:
            bound = input_row[first_pattern.variable]
            starts = [bound] if isinstance(bound, Node) else []
        else:
            starts = list(graph.nodes.values())
        matches: list[dict[str, Any]] = []
        for start in starts:
            if not _gql_pattern_node_matches(start, first_pattern, context):
                continue
            states = [
                (dict(input_row, **{first_pattern.variable: start}), PathValue((start.id,), ()))
            ]
            for edge_index in range(len(pattern.edges)):
                next_states: list[tuple[dict[str, Any], PathValue]] = []
                for state, prefix in states:
                    current = state.get(pattern.nodes[edge_index].variable)
                    if isinstance(current, Node):
                        for candidate, suffix in _gql_expand_pattern(
                            graph, current, pattern, edge_index, state, context
                        ):
                            next_states.append(
                                (
                                    candidate,
                                    PathValue(
                                        prefix.nodes + suffix.nodes[1:], prefix.edges + suffix.edges
                                    ),
                                )
                            )
                states = next_states
            for candidate, path in states:
                if pattern.path_variable:
                    if (
                        pattern.path_variable in candidate
                        and candidate[pattern.path_variable] != path
                    ):
                        continue
                    candidate[pattern.path_variable] = path
                matches.append(
                    {
                        key: value
                        for key, value in candidate.items()
                        if key not in {n.variable for n in pattern.nodes if n.anonymous}
                    }
                )
        if matches:
            output.extend(matches)
        elif clause.optional:
            padded = dict(input_row)
            for variable in pattern.variables - {n.variable for n in pattern.nodes if n.anonymous}:
                padded.setdefault(variable, None)
            output.append(padded)
    return output


def _eval_gql_expr(
    expr: Expr,
    graph: PropertyGraph,
    context: PolicyEvaluationContext,
    row: dict[str, Any],
) -> Any:
    context.tick()
    if isinstance(expr, Literal):
        return expr.value
    if isinstance(expr, VariableRef):
        if expr.name.startswith("$"):
            return parameter(context.parameters, expr.name[1:])
        if expr.name.upper() == "SESSION_USER":
            return context.user
        return row.get(expr.name)
    if isinstance(expr, PropertyRef):
        resource = row.get(expr.variable)
        return (
            resource.properties.get(expr.property_name)
            if isinstance(resource, (Node, Edge))
            else None
        )
    if isinstance(expr, LabelTest):
        resource = row.get(expr.variable)
        if isinstance(resource, Node):
            return expr.label in resource.labels
        return (
            None
            if resource is None
            else isinstance(resource, Edge) and expr.label in resource.labels
        )
    if isinstance(expr, UnaryExpr):
        value = _eval_gql_expr(expr.operand, graph, context, row)
        if expr.op == "NOT":
            return _truth_not(value)
        if expr.op == "IS NULL":
            return value is None
        if expr.op == "IS NOT NULL":
            return value is not None
        raise CatalogError(f"Unsupported policy unary operator: {expr.op}")
    if isinstance(expr, BinaryExpr):
        if expr.op == "AND":
            return _truth_and(
                _eval_gql_expr(expr.left, graph, context, row),
                _eval_gql_expr(expr.right, graph, context, row),
            )
        if expr.op == "OR":
            return _truth_or(
                _eval_gql_expr(expr.left, graph, context, row),
                _eval_gql_expr(expr.right, graph, context, row),
            )
        left = _eval_gql_expr(expr.left, graph, context, row)
        if expr.op == "IN" and isinstance(expr.right, FunctionCall):
            right = [_eval_gql_expr(item, graph, context, row) for item in expr.right.args]
        else:
            right = _eval_gql_expr(expr.right, graph, context, row)
        return _compare(left, expr.op, right)
    if isinstance(expr, FunctionCall):
        name = expr.name.upper()
        values = [_eval_gql_expr(item, graph, context, row) for item in expr.args]
        if name == "LIST":
            return values
        if name in {"ID", "ELEMENT_ID"}:
            return (
                values[0].id if len(values) == 1 and isinstance(values[0], (Node, Edge)) else None
            )
        if name == "COALESCE":
            return next((value for value in values if value is not None), None)
        if name == "PARAM":
            if len(values) != 1 or not isinstance(values[0], str):
                raise CatalogError("PARAM expects one string argument")
            return parameter(context.parameters, values[0])
        if name == "HAS_ROLE":
            if len(values) != 1 or not isinstance(values[0], str):
                raise CatalogError("HAS_ROLE expects a string")
            return values[0] in context.roles
        if name == "PROPERTY_EXISTS":
            if len(values) != 2 or not isinstance(values[1], str):
                raise CatalogError("PROPERTY_EXISTS expects an element and a string")
            if values[0] is not None and not isinstance(values[0], (Node, Edge)):
                raise CatalogError("PROPERTY_EXISTS expects an element and a string")
            return (
                len(values) == 2
                and isinstance(values[0], (Node, Edge))
                and str(values[1]) in values[0].properties
            )
        raise CatalogError(f"Unsupported policy function: {expr.name}")
    if isinstance(expr, ExistsExpr):
        rows = [dict(row)]
        for clause in expr.query.matches:
            rows = _gql_match_clause(graph, clause, rows, context)
        if expr.query.where is not None:
            rows = [
                item
                for item in rows
                if _eval_gql_expr(expr.query.where, graph, context, item) is True
            ]
        return bool(rows)
    raise CatalogError(f"Unsupported GQL policy expression: {expr!r}")


def compile_gql_predicate(text: str) -> Predicate:
    """Compile the side-effect-free GQL expression subset used by policies."""

    if len(text) > 8192:
        raise CatalogError("Policy source exceeds 8192 characters")
    try:
        expression = parse_expression(text.strip())
    except Exception as exc:
        raise CatalogError(f"Invalid GQL policy predicate: {exc}") from exc

    nodes = list(walk_expr(expression))
    try:
        require(
            expression_type(
                expression, {"RESOURCE": "ELEMENT", "NEW_RESOURCE": "ELEMENT"}, policy=True
            ),
            {"BOOL"},
        )
    except Exception as exc:
        raise CatalogError("Invalid policy names, types or function arguments") from exc
    if len(nodes) > 256:
        raise CatalogError("Policy expression exceeds 256 AST nodes")
    # Bounding total EXISTS nodes also bounds their nesting depth.
    if sum(isinstance(n, ExistsExpr) for n in nodes) > 4:
        raise CatalogError("Policy expression exceeds four EXISTS subqueries")
    allowed_nodes = (
        Literal,
        VariableRef,
        PropertyRef,
        LabelTest,
        UnaryExpr,
        BinaryExpr,
        FunctionCall,
        ExistsExpr,
    )
    for item in nodes:
        if not isinstance(item, allowed_nodes):
            raise CatalogError("Policy AST node is not admissible")
        if isinstance(item, FunctionCall) and item.name.upper() not in {
            "LIST",
            "COALESCE",
            "PARAM",
            "HAS_ROLE",
            "PROPERTY_EXISTS",
        }:
            raise CatalogError("Policy function is not admissible")
        if not isinstance(item, ExistsExpr):
            continue
        query = item.query
        if (
            query.update_kind is not None
            or query.return_items
            or query.order_by
            or query.limit is not None
            or any(clause.optional for clause in query.matches)
            or any(
                keyword not in {"MATCH", "WHERE"}
                for keyword, _content in scan_clauses(query.source)
            )
        ):
            raise CatalogError("Policy EXISTS supports only MATCH clauses with an optional WHERE")
        if sum(len(c.pattern.nodes) + len(c.pattern.edges) for c in query.matches) > 64:
            raise CatalogError("Policy EXISTS exceeds 64 pattern constituents")

    def predicate(graph: PropertyGraph, context: PolicyEvaluationContext) -> TruthValue:
        bindings = {
            "RESOURCE": context.resource,
            "NEW_RESOURCE": context.new_resource,
        }
        return _as_truth(_eval_gql_expr(expression, graph, context, bindings))

    return predicate


def compile_predicate(spec: Any, _depth: int = 0) -> Predicate:
    """Compile the JSON policy predicate language into a pure callable.

    The prototype intentionally uses a small, deterministic JSON AST instead of
    evaluating Python expressions. It covers boolean composition, resource labels,
    property comparisons, context parameters, and bounded relationship tests.
    """

    if _depth > 16:
        raise CatalogError("JSON policy exceeds 16 levels")
    if isinstance(spec, dict) and len(str(spec)) > 8192:
        raise CatalogError("JSON policy exceeds source budget")
    if spec is None or spec is True:
        return lambda graph, context: True
    if spec is False:
        return lambda graph, context: False
    if isinstance(spec, str):
        return compile_gql_predicate(spec)
    if not isinstance(spec, dict):
        raise CatalogError(f"Policy predicate must be a JSON object or Boolean, got {spec!r}")
    if len(spec) != 1:
        raise CatalogError("A JSON predicate must contain exactly one operator")

    if "all" in spec:
        if not isinstance(spec["all"], list):
            raise CatalogError("JSON all requires a list of predicates")
        compiled_all = [compile_predicate(item, _depth + 1) for item in spec["all"]]

        def all_predicates(graph: PropertyGraph, context: PolicyEvaluationContext) -> TruthValue:
            result: TruthValue = True
            for predicate in compiled_all:
                result = _truth_and(result, predicate(graph, context))
            return result

        return all_predicates
    if "any" in spec:
        if not isinstance(spec["any"], list):
            raise CatalogError("JSON any requires a list of predicates")
        compiled_any = [compile_predicate(item, _depth + 1) for item in spec["any"]]

        def any_predicates(graph: PropertyGraph, context: PolicyEvaluationContext) -> TruthValue:
            result: TruthValue = False
            for predicate in compiled_any:
                result = _truth_or(result, predicate(graph, context))
            return result

        return any_predicates
    if "not" in spec:
        compiled_not = compile_predicate(spec["not"], _depth + 1)
        return lambda graph, context: _truth_not(compiled_not(graph, context))
    if "resource_label" in spec:
        label = str(spec["resource_label"])
        return lambda graph, context: (
            isinstance(context.resource, Node) and label in context.resource.labels
        )
    if "new_resource_label" in spec:
        label = str(spec["new_resource_label"])
        return lambda graph, context: (
            isinstance(context.new_resource, Node) and label in context.new_resource.labels
        )
    if "resource_type" in spec:
        edge_type = str(spec["resource_type"])
        return lambda graph, context: (
            isinstance(context.resource, Edge) and edge_type in context.resource.labels
        )
    if "property_compare" in spec:
        item = spec["property_compare"]
        source_name = str(item.get("source", "RESOURCE"))
        property_name = str(item["property"])
        op = str(item.get("op", "="))
        expected = item.get("value")

        def property_compare(graph: PropertyGraph, context: PolicyEvaluationContext) -> TruthValue:
            source = _source(context, source_name)
            if not isinstance(source, (Node, Edge)):
                return None
            right = expected
            if expected == "SESSION_USER":
                right = context.user
            elif isinstance(expected, str) and expected.startswith("PARAM:"):
                right = parameter(context.parameters, expected.split(":", 1)[1])
            return _compare(source.properties.get(property_name), op, right)

        return property_compare
    if "context_compare" in spec:
        item = spec["context_compare"]
        name = str(item["name"])
        op = str(item.get("op", "="))
        expected = item.get("value")
        return lambda graph, context: _compare(parameter(context.parameters, name), op, expected)
    if "path_exists" in spec:
        path_spec = spec["path_exists"]
        if len(path_spec.get("steps", [])) > 12:
            raise CatalogError("JSON policy path exceeds 12 steps")
        return lambda graph, context: _path_exists(graph, path_spec, context)

    raise CatalogError(f"Unsupported policy predicate form: {spec}")
