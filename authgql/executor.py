from __future__ import annotations

from dataclasses import dataclass
from functools import cmp_to_key
from time import perf_counter
from typing import Any, Iterable
import copy
import json
import uuid

from .analyzer import StaticAnalyzer
from .authorization import AuthorizationEngine
from .catalog import AuthorizationCatalog
from .errors import ExecutionError
from .metrics import ExecutionMetrics
from .model import Edge, GraphElement, Node, PropertyGraph
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
    PatternEdge,
    PatternNode,
    PropertyRef,
    Query,
    UnaryExpr,
    VariableRef,
)


@dataclass
class ExecutionResult:
    columns: list[str]
    rows: list[dict[str, Any]]
    metrics: ExecutionMetrics
    updated: bool = False
    affected_elements: int = 0
    reference_equal: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": "ok",
            "columns": self.columns,
            "rows": self.rows,
            "updated": self.updated,
            "affected_elements": self.affected_elements,
            "metrics": self.metrics.to_dict(),
        }
        if self.reference_equal is not None:
            result["reference_equal"] = self.reference_equal
        return result


class SecureExecutor:
    def __init__(
        self,
        graph: PropertyGraph,
        catalog: AuthorizationCatalog,
        user: str,
        parameters: dict[str, Any] | None = None,
        preflight: bool = True,
    ) -> None:
        self.graph = graph
        self.catalog = catalog
        self.user = user
        self.parameters = parameters or {}
        self.preflight_enabled = preflight
        self.metrics = ExecutionMetrics()
        self.authorization = AuthorizationEngine(
            catalog, graph, user, self.parameters, self.metrics
        )

    def execute(self, query: Query, compare_reference: bool = False) -> ExecutionResult:
        start = perf_counter()
        analyzer = StaticAnalyzer(
            self.catalog, self.graph.name, self.user, metrics=self.metrics
        )
        if self.preflight_enabled:
            analyzer.preflight(query)
        else:
            analyzer.derive_obligations(query)
        self.authorization.check_access()

        if query.update_kind:
            result = self._execute_update(query)
        else:
            rows = self._evaluate_query_rows(query)
            projected = self._project(query, rows)
            result = ExecutionResult(
                columns=[item.alias for item in query.return_items],
                rows=projected,
                metrics=self.metrics,
            )
        self.metrics.execution_ms += (perf_counter() - start) * 1000.0

        if compare_reference:
            if query.update_kind:
                raise ExecutionError("Reference comparison is supported only for read-only queries")
            reference = self._execute_reference(query)
            result.reference_equal = _canonical_rows(result.rows) == _canonical_rows(reference.rows)
            if not result.reference_equal:
                raise ExecutionError(
                    "Secure execution differs from the internal materialized-view cross-check"
                )
        return result

    # ---------- Pattern evaluation ----------

    def _evaluate_query_rows(
        self,
        query: Query,
        initial_rows: list[dict[str, Any]] | None = None,
        graph: PropertyGraph | None = None,
    ) -> list[dict[str, Any]]:
        active_graph = graph or self.graph
        rows = initial_rows if initial_rows is not None else [{}]
        for match in query.matches:
            rows = self._apply_match(match, rows, active_graph)
        if query.where is not None:
            filtered: list[dict[str, Any]] = []
            for row in rows:
                if _truthy(self._eval_expr(query.where, row, active_graph)):
                    filtered.append(row)
            rows = filtered
        if query.order_by:
            rows = self._order_rows(rows, query, active_graph)
        if query.limit is not None:
            rows = rows[: query.limit]
        return rows

    def _apply_match(
        self,
        clause: MatchClause,
        input_rows: list[dict[str, Any]],
        graph: PropertyGraph,
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        introduced = clause.pattern.variables
        for input_row in input_rows:
            matches = self._match_chain(clause.pattern, input_row, graph)
            if matches:
                output.extend(matches)
            elif clause.optional:
                padded = dict(input_row)
                for variable in introduced:
                    padded.setdefault(variable, None)
                output.append(padded)
        return output

    def _match_chain(
        self,
        pattern: PatternChain,
        base_row: dict[str, Any],
        graph: PropertyGraph,
    ) -> list[dict[str, Any]]:
        first = pattern.nodes[0]
        initial: list[tuple[dict[str, Any], Node]] = []
        if first.variable in base_row and base_row[first.variable] is not None:
            candidate = base_row[first.variable]
            if isinstance(candidate, Node) and self._admit_node(candidate, first, base_row, graph):
                initial.append((dict(base_row), candidate))
        else:
            for candidate in graph.nodes.values():
                self.metrics.node_scan_candidates += 1
                if self._admit_node(candidate, first, base_row, graph):
                    row = dict(base_row)
                    row[first.variable] = candidate
                    self.metrics.nodes_enqueued += 1
                    initial.append((row, candidate))

        states = initial
        for edge_pattern, node_pattern in zip(pattern.edges, pattern.nodes[1:]):
            next_states: list[tuple[dict[str, Any], Node]] = []
            for row, current_node in states:
                for path_edges, end_node in self._expand(
                    current_node, edge_pattern, row, node_pattern, graph
                ):
                    if node_pattern.variable in row and row[node_pattern.variable] is not None:
                        bound = row[node_pattern.variable]
                        if not isinstance(bound, Node) or bound.id != end_node.id:
                            continue
                    new_row = dict(row)
                    new_row[node_pattern.variable] = end_node
                    if edge_pattern.variable:
                        new_row[edge_pattern.variable] = (
                            path_edges[0]
                            if edge_pattern.min_hops == edge_pattern.max_hops == 1
                            else list(path_edges)
                        )
                    self.metrics.bindings_materialized += 1
                    next_states.append((new_row, end_node))
            states = next_states
            if not states:
                break
        return [row for row, _ in states]

    def _admit_node(
        self,
        candidate: Node,
        pattern: PatternNode,
        row: dict[str, Any],
        graph: PropertyGraph,
    ) -> bool:
        if pattern.labels and not pattern.labels.issubset(candidate.labels):
            return False
        if not self.authorization.permits("TRAVERSE", candidate, graph=graph):
            self.metrics.denied_before_enqueue += 1
            return False
        for property_name, expected in pattern.properties.items():
            actual = self.authorization.read_property(candidate, property_name)
            if _compare(actual, "=", self._resolve_pattern_value(expected)) is not True:
                return False
        return True

    def _admit_edge(
        self,
        edge: Edge,
        pattern: PatternEdge,
        graph: PropertyGraph,
    ) -> bool:
        if pattern.edge_type and edge.type != pattern.edge_type:
            return False
        if not self.authorization.permits("TRAVERSE", edge, graph=graph):
            self.metrics.denied_before_enqueue += 1
            return False
        for property_name, expected in pattern.properties.items():
            actual = self.authorization.read_property(edge, property_name)
            if _compare(actual, "=", self._resolve_pattern_value(expected)) is not True:
                return False
        return True

    def _expand(
        self,
        start: Node,
        edge_pattern: PatternEdge,
        row: dict[str, Any],
        end_pattern: PatternNode,
        graph: PropertyGraph,
    ) -> Iterable[tuple[list[Edge], Node]]:
        if edge_pattern.min_hops == 0:
            if self._admit_node(start, end_pattern, row, graph):
                yield [], start

        # The supported syntax has no explicit path-mode prefix, so GQL's
        # default WALK behavior applies. The parser's finite maximum (12) keeps
        # quantified expansion bounded even when nodes or edges repeat.
        frontier: list[tuple[Node, list[Edge]]] = [(start, [])]
        while frontier:
            current, path = frontier.pop()
            if len(path) >= edge_pattern.max_hops:
                continue
            for edge, neighbour in graph.adjacent(current.id, edge_pattern.direction):
                self.metrics.edge_candidates += 1
                if not self._admit_edge(edge, edge_pattern, graph):
                    continue
                # Endpoint authorization occurs before enqueueing the candidate.
                if not self.authorization.permits("TRAVERSE", neighbour, graph=graph):
                    self.metrics.denied_before_enqueue += 1
                    continue
                self.metrics.edges_enqueued += 1
                self.metrics.nodes_enqueued += 1
                new_path = [*path, edge]
                if len(new_path) >= edge_pattern.min_hops and self._admit_node(
                    neighbour, end_pattern, row, graph
                ):
                    yield new_path, neighbour
                if len(new_path) < edge_pattern.max_hops:
                    frontier.append((neighbour, new_path))

    # ---------- Expressions and result shaping ----------

    def _eval_expr(self, expr: Expr, row: dict[str, Any], graph: PropertyGraph) -> Any:
        if isinstance(expr, Literal):
            return expr.value
        if isinstance(expr, VariableRef):
            upper = expr.name.upper()
            if upper == "SESSION_USER":
                return self.user
            return row.get(expr.name)
        if isinstance(expr, PropertyRef):
            resource = row.get(expr.variable)
            if resource is None:
                return None
            if not isinstance(resource, (Node, Edge)):
                raise ExecutionError(f"{expr.variable!r} is not a graph element")
            return self.authorization.read_property(
                resource, expr.property_name, graph=graph
            )
        if isinstance(expr, LabelTest):
            resource = row.get(expr.variable)
            return isinstance(resource, Node) and expr.label in resource.labels
        if isinstance(expr, UnaryExpr):
            value = self._eval_expr(expr.operand, row, graph)
            if expr.op == "NOT":
                return _gql_not(value)
            if expr.op == "IS NULL":
                return value is None
            if expr.op == "IS NOT NULL":
                return value is not None
            raise ExecutionError(f"Unsupported unary operator: {expr.op}")
        if isinstance(expr, BinaryExpr):
            if expr.op == "AND":
                left = self._eval_expr(expr.left, row, graph)
                if left is False:
                    return False
                return _gql_and(left, self._eval_expr(expr.right, row, graph))
            if expr.op == "OR":
                left = self._eval_expr(expr.left, row, graph)
                if left is True:
                    return True
                return _gql_or(left, self._eval_expr(expr.right, row, graph))
            left = self._eval_expr(expr.left, row, graph)
            if expr.op == "IN" and isinstance(expr.right, FunctionCall):
                right = [self._eval_expr(item, row, graph) for item in expr.right.args]
            else:
                right = self._eval_expr(expr.right, row, graph)
            return _compare(left, expr.op, right)
        if isinstance(expr, FunctionCall):
            name = expr.name.upper()
            if name == "LIST":
                return [self._eval_expr(arg, row, graph) for arg in expr.args]
            if name in {"ID", "ELEMENT_ID"}:
                if len(expr.args) != 1:
                    raise ExecutionError(f"{name} expects one argument")
                resource = self._eval_expr(expr.args[0], row, graph)
                return resource.id if isinstance(resource, (Node, Edge)) else None
            if name == "COALESCE":
                for arg in expr.args:
                    value = self._eval_expr(arg, row, graph)
                    if value is not None:
                        return value
                return None
            if name == "COUNT":
                raise ExecutionError("COUNT is evaluated during aggregation")
            raise ExecutionError(f"Unsupported function: {name}")
        if isinstance(expr, ExistsExpr):
            nested_rows = self._evaluate_query_rows(expr.query, [dict(row)], graph)
            return bool(nested_rows)
        raise ExecutionError(f"Unsupported expression: {expr!r}")

    def _project(
        self,
        query: Query,
        rows: list[dict[str, Any]],
        graph: PropertyGraph | None = None,
    ) -> list[dict[str, Any]]:
        active_graph = graph or self.graph
        aggregate = any(_contains_aggregate(item.expression) for item in query.return_items)
        if aggregate:
            if not all(_contains_aggregate(item.expression) for item in query.return_items):
                raise ExecutionError(
                    "The prototype requires every RETURN item to be aggregate when COUNT is used"
                )
            projected: dict[str, Any] = {}
            for item in query.return_items:
                projected[item.alias] = self._eval_aggregate(
                    item.expression, rows, active_graph
                )
            return [projected]

        output: list[dict[str, Any]] = []
        for row in rows:
            result_row = {
                item.alias: _json_value(
                    self._eval_expr(item.expression, row, active_graph)
                )
                for item in query.return_items
            }
            output.append(result_row)
        return output

    def _eval_aggregate(
        self,
        expr: Expr,
        rows: list[dict[str, Any]],
        graph: PropertyGraph,
    ) -> Any:
        if not isinstance(expr, FunctionCall) or expr.name.upper() != "COUNT":
            raise ExecutionError("Only COUNT aggregates are implemented")
        if len(expr.args) != 1:
            raise ExecutionError("COUNT expects one argument")
        argument = expr.args[0]
        if isinstance(argument, VariableRef) and argument.name == "*":
            return len(rows)
        return sum(
            1 for row in rows if self._eval_expr(argument, row, graph) is not None
        )

    def _order_rows(
        self, rows: list[dict[str, Any]], query: Query, graph: PropertyGraph
    ) -> list[dict[str, Any]]:
        def compare_rows(left_row: dict[str, Any], right_row: dict[str, Any]) -> int:
            for item in query.order_by:
                left = self._eval_expr(item.expression, left_row, graph)
                right = self._eval_expr(item.expression, right_row, graph)
                comparison = _safe_compare(left, right)
                if comparison:
                    return -comparison if item.descending else comparison
            return 0

        return sorted(rows, key=cmp_to_key(compare_rows))

    # ---------- Transactional updates ----------

    def _execute_update(self, query: Query) -> ExecutionResult:
        if query.update_kind == "INSERT":
            return self._execute_insert(query)

        rows = self._evaluate_query_rows(query)
        prospective = self.graph.clone()
        affected: set[str] = set()
        return_rows = rows

        if query.update_kind == "SET":
            staged: dict[str, dict[str, Any]] = {}
            original_by_id: dict[str, GraphElement] = {}
            for row in rows:
                for assignment in query.set_assignments:
                    resource = row.get(assignment.variable)
                    if not isinstance(resource, (Node, Edge)):
                        raise ExecutionError(
                            f"SET variable {assignment.variable!r} is not bound to a graph element"
                        )
                    self.metrics.update_candidates += 1
                    self.authorization.check("SET", resource, assignment.property_name)
                    value = self._eval_expr(assignment.expression, row, self.graph)
                    staged.setdefault(resource.id, {})[assignment.property_name] = copy.deepcopy(value)
                    original_by_id[resource.id] = resource

            for element_id, changes in staged.items():
                clone_element = prospective.element(element_id)
                clone_element.properties.update(changes)
            for element_id, changes in staged.items():
                old = original_by_id[element_id]
                new = prospective.element(element_id)
                for property_name in changes:
                    self.authorization.check_with(
                        "SET", old, new, prospective, property_name=property_name
                    )
                affected.add(element_id)

            return_rows = [
                {
                    key: (prospective.element(value.id) if isinstance(value, (Node, Edge)) and value.id in prospective.nodes | prospective.edges else value)
                    for key, value in row.items()
                }
                for row in rows
            ]

        elif query.update_kind == "REMOVE":
            staged_removals: dict[str, set[str]] = {}
            removal_originals: dict[str, GraphElement] = {}
            for row in rows:
                for remove_assignment in query.remove_assignments:
                    resource = row.get(remove_assignment.variable)
                    if not isinstance(resource, (Node, Edge)):
                        raise ExecutionError(
                            f"REMOVE variable {remove_assignment.variable!r} is not bound to a graph element"
                        )
                    self.metrics.update_candidates += 1
                    self.authorization.check(
                        "REMOVE", resource, remove_assignment.property_name
                    )
                    staged_removals.setdefault(resource.id, set()).add(
                        remove_assignment.property_name
                    )
                    removal_originals[resource.id] = resource

            for element_id, property_names in staged_removals.items():
                clone_element = prospective.element(element_id)
                for property_name in property_names:
                    clone_element.properties.pop(property_name, None)
            for element_id, property_names in staged_removals.items():
                old = removal_originals[element_id]
                new = prospective.element(element_id)
                for property_name in property_names:
                    self.authorization.check_with(
                        "REMOVE", old, new, prospective, property_name=property_name
                    )
                affected.add(element_id)

            return_rows = [
                {
                    key: (
                        prospective.element(value.id)
                        if isinstance(value, (Node, Edge))
                        and value.id in prospective.nodes | prospective.edges
                        else value
                    )
                    for key, value in row.items()
                }
                for row in rows
            ]

        elif query.update_kind == "DELETE":
            to_delete: dict[str, GraphElement] = {}
            for row in rows:
                for variable in query.delete_variables:
                    resource = row.get(variable)
                    if not isinstance(resource, (Node, Edge)):
                        raise ExecutionError(
                            f"DELETE variable {variable!r} is not bound to a graph element"
                        )
                    to_delete[resource.id] = resource
            explicitly_deleted = set(to_delete)

            # Element-level authorization is part of the request check and must
            # precede ordinary NODETACH validation. Otherwise G1001 could reveal
            # incident topology for a resource whose DELETE policy denies access.
            authorized_for_delete: set[str] = set()
            for resource in to_delete.values():
                self.metrics.update_candidates += 1
                self.authorization.check("DELETE", resource)
                authorized_for_delete.add(resource.id)

            for resource in list(to_delete.values()):
                if not isinstance(resource, Node):
                    continue
                incident_ids = [edge.id for edge in self.graph.out_edges(resource.id)] + [
                    edge.id for edge in self.graph.in_edges(resource.id)
                ]
                if query.delete_detach:
                    for edge_id in dict.fromkeys(incident_ids):
                        to_delete.setdefault(edge_id, self.graph.edges[edge_id])
                else:
                    missing = set(incident_ids).difference(explicitly_deleted)
                    if missing:
                        raise ExecutionError(
                            "NODETACH DELETE requires every incident edge to be explicitly deleted",
                            code="G1001",
                        )

            for resource in to_delete.values():
                if resource.id not in authorized_for_delete:
                    self.metrics.update_candidates += 1
                    self.authorization.check("DELETE", resource)
                    authorized_for_delete.add(resource.id)
            for resource in to_delete.values():
                if isinstance(resource, Edge) and resource.id in prospective.edges:
                    prospective.delete_edge(resource.id)
                affected.add(resource.id)
            for resource in to_delete.values():
                if isinstance(resource, Node) and resource.id in prospective.nodes:
                    prospective.delete_node(resource.id)
                    affected.add(resource.id)
        else:
            raise ExecutionError(f"Unsupported update kind: {query.update_kind}")

        projected = (
            self._project(query, return_rows, prospective)
            if query.return_items
            else []
        )
        self.graph.replace_with(prospective)
        return ExecutionResult(
            columns=[item.alias for item in query.return_items],
            rows=projected,
            metrics=self.metrics,
            updated=True,
            affected_elements=len(affected),
        )

    def _execute_insert(self, query: Query) -> ExecutionResult:
        prospective = self.graph.clone()
        source_rows = self._evaluate_query_rows(query) if query.matches else [{}]
        result_rows: list[dict[str, Any]] = []
        inserted: dict[str, GraphElement] = {}
        existing_endpoints: dict[str, GraphElement] = {}

        for source_row in source_rows:
            row = dict(source_row)
            for pattern in query.insert_patterns:
                pattern_nodes: list[Node] = []
                for node_spec in pattern.nodes:
                    bound = row.get(node_spec.variable)
                    if bound is not None:
                        if not isinstance(bound, Node):
                            raise ExecutionError(
                                f"INSERT endpoint {node_spec.variable!r} is not a node"
                            )
                        node = prospective.nodes.get(bound.id)
                        if node is None:
                            raise ExecutionError(
                                f"INSERT endpoint {bound.id!r} is absent from the prospective graph"
                            )
                        if node_spec.labels and not node_spec.labels.issubset(node.labels):
                            raise ExecutionError(
                                f"Bound INSERT endpoint {bound.id!r} does not satisfy its labels"
                            )
                        if node.id not in inserted:
                            existing_endpoints[node.id] = self.graph.nodes[node.id]
                    else:
                        element_id = str(
                            node_spec.properties.get("_id") or f"n_{uuid.uuid4().hex[:12]}"
                        )
                        properties = {
                            key: self._resolve_pattern_value(value)
                            for key, value in node_spec.properties.items()
                            if key != "_id"
                        }
                        node = Node(element_id, set(node_spec.labels), properties)
                        prospective.add_node(node)
                        inserted[node.id] = node
                        row[node_spec.variable] = node
                    pattern_nodes.append(node)

                for index, edge_spec in enumerate(pattern.edges):
                    if edge_spec.min_hops != 1 or edge_spec.max_hops != 1:
                        raise ExecutionError("INSERT does not accept quantified edge patterns")
                    edge_id = str(
                        edge_spec.properties.get("_id") or f"e_{uuid.uuid4().hex[:12]}"
                    )
                    properties = {
                        key: self._resolve_pattern_value(value)
                        for key, value in edge_spec.properties.items()
                        if key != "_id"
                    }
                    left = pattern_nodes[index]
                    right = pattern_nodes[index + 1]
                    if edge_spec.direction == "in":
                        source, target = right, left
                    elif edge_spec.direction == "out":
                        source, target = left, right
                    else:
                        raise ExecutionError(
                            "The prototype requires a direction for inserted edges"
                        )
                    edge = Edge(
                        edge_id,
                        edge_spec.edge_type or "EDGE",
                        source.id,
                        target.id,
                        properties,
                    )
                    prospective.add_edge(edge)
                    inserted[edge.id] = edge
                    if edge_spec.variable:
                        row[edge_spec.variable] = edge
            result_rows.append(row)

        # The entire delta exists before any WITH CHECK predicate is evaluated,
        # allowing a policy to inspect another element inserted by the statement.
        for element in inserted.values():
            self.metrics.update_candidates += 1
            self.authorization.check_with("INSERT", None, element, prospective)
        for endpoint in existing_endpoints.values():
            self.authorization.check("TRAVERSE", endpoint)

        projected = (
            self._project(query, result_rows, prospective)
            if query.return_items
            else []
        )
        self.graph.replace_with(prospective)
        return ExecutionResult(
            columns=[item.alias for item in query.return_items],
            rows=projected,
            metrics=self.metrics,
            updated=True,
            affected_elements=len(inserted),
        )

    # ---------- Internal materialized-view cross-check ----------

    def _execute_reference(self, query: Query) -> ExecutionResult:
        # Materialize the authorized topology, then execute the same query on that view.
        # This internal cross-check shares major execution components. It is neither
        # independent validation nor a proof of storage-level traversal-touch safety.
        reference_metrics = ExecutionMetrics()
        reference_auth = AuthorizationEngine(
            self.catalog, self.graph, self.user, self.parameters, reference_metrics
        )
        allowed_nodes = {
            node_id: node.clone()
            for node_id, node in self.graph.nodes.items()
            if reference_auth.permits("TRAVERSE", node)
        }
        allowed_edges = {
            edge_id: edge.clone()
            for edge_id, edge in self.graph.edges.items()
            if edge.source in allowed_nodes
            and edge.target in allowed_nodes
            and reference_auth.permits("TRAVERSE", edge)
        }
        view = PropertyGraph(self.graph.name, allowed_nodes, allowed_edges)
        executor = SecureExecutor(
            view,
            self.catalog,
            self.user,
            self.parameters,
            preflight=self.preflight_enabled,
        )
        return executor.execute(query, compare_reference=False)

    def _resolve_pattern_value(self, value: Any) -> Any:
        if isinstance(value, ContextValue):
            if value.name == "SESSION_USER":
                return self.user
        return value


def _contains_aggregate(expr: Expr) -> bool:
    if isinstance(expr, FunctionCall):
        return expr.name.upper() == "COUNT" or any(_contains_aggregate(arg) for arg in expr.args)
    if isinstance(expr, UnaryExpr):
        return _contains_aggregate(expr.operand)
    if isinstance(expr, BinaryExpr):
        return _contains_aggregate(expr.left) or _contains_aggregate(expr.right)
    return False


def _compare(left: Any, op: str, right: Any) -> bool | None:
    if op == "IN":
        if left is None or right is None:
            return None
        if not isinstance(right, list):
            raise ExecutionError("IN expects a list on the right-hand side")
        saw_unknown = False
        for item in right:
            comparison = _compare(left, "=", item)
            if comparison is True:
                return True
            saw_unknown = saw_unknown or comparison is None
        return None if saw_unknown else False
    if left is None or right is None:
        return None
    if op in {"=", "=="}:
        return left == right
    if op in {"<>", "!="}:
        return left != right
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    if op == ">":
        return left > right
    if op == ">=":
        return left >= right
    raise ExecutionError(f"Unsupported comparison operator: {op}")


def _truthy(value: Any) -> bool:
    return value is True


def _gql_not(value: Any) -> bool | None:
    if value is None:
        return None
    return not bool(value)


def _gql_and(left: Any, right: Any) -> bool | None:
    if left is False or right is False:
        return False
    if left is None or right is None:
        return None
    return bool(left) and bool(right)


def _gql_or(left: Any, right: Any) -> bool | None:
    if left is True or right is True:
        return True
    if left is None or right is None:
        return None
    return bool(left) or bool(right)


def _safe_compare(left: Any, right: Any) -> int:
    if left is None and right is None:
        return 0
    if left is None:
        return 1
    if right is None:
        return -1
    if left < right:
        return -1
    if left > right:
        return 1
    return 0


def _json_value(value: Any) -> Any:
    if isinstance(value, Node):
        # GQL returns a graph-element reference value. Serializing the backing
        # property map here would bypass explicit READ guards on ``n.property``.
        return {
            "id": value.id,
            "kind": "node",
        }
    if isinstance(value, Edge):
        return {
            "id": value.id,
            "kind": "edge",
        }
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return copy.deepcopy(value)


def _canonical_rows(rows: list[dict[str, Any]]) -> list[str]:
    return sorted(json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows)
