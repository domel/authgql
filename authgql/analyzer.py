from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from .catalog import AuthorizationCatalog, Target
from .errors import AuthorizationError
from .metrics import ExecutionMetrics
from .validation import validate_query
from .parser import (
    BinaryExpr,
    ExistsExpr,
    Expr,
    FunctionCall,
    PatternChain,
    PropertyRef,
    Query,
    UnaryExpr,
)


@dataclass(frozen=True)
class Obligation:
    action: str
    target: Target
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "target": self.target.to_dict(),
            "reason": self.reason,
        }


class StaticAnalyzer:
    def __init__(
        self,
        catalog: AuthorizationCatalog,
        graph_name: str,
        user: str,
        metrics: ExecutionMetrics | None = None,
    ) -> None:
        self.catalog = catalog
        self.graph_name = graph_name
        self.user = user
        self.metrics = metrics or ExecutionMetrics()

    def derive_obligations(self, query: Query) -> list[Obligation]:
        start = perf_counter()
        validate_query(query)
        obligations: list[Obligation] = [
            Obligation("ACCESS", Target("GRAPH", self.graph_name), "resolve the working graph")
        ]
        variable_nodes: dict[str, set[str]] = {}
        variable_edges: dict[str, set[str]] = {}

        def add_pattern(
            pattern: PatternChain,
            reason_prefix: str,
            node_scope: dict[str, set[str]] | None = None,
            edge_scope: dict[str, set[str]] | None = None,
        ) -> None:
            node_scope = variable_nodes if node_scope is None else node_scope
            edge_scope = variable_edges if edge_scope is None else edge_scope
            for node in pattern.nodes:
                node_scope.setdefault(node.variable, set()).update(node.labels)
                target = (
                    Target("NODES", self.graph_name, labels=frozenset(node.labels))
                    if node.labels
                    else Target("GRAPH", self.graph_name)
                )
                obligations.append(
                    Obligation("TRAVERSE", target, f"{reason_prefix}: node pattern {node.variable}")
                )
                for property_name in node.properties:
                    obligations.append(
                        Obligation(
                            "READ",
                            Target(
                                "PROPERTIES",
                                self.graph_name,
                                labels=frozenset(node.labels),
                                properties=frozenset({property_name}),
                            ),
                            f"{reason_prefix}: node pattern property {node.variable}.{property_name}",
                        )
                    )
            for edge in pattern.edges:
                if edge.variable:
                    edge_scope.setdefault(edge.variable, set()).update(edge.required_labels)
                target = (
                    Target("EDGES", self.graph_name, edge_types=frozenset(edge.required_labels))
                    if edge.required_labels
                    else Target("GRAPH", self.graph_name)
                )
                obligations.append(
                    Obligation("TRAVERSE", target, f"{reason_prefix}: edge expansion")
                )
                for property_name in edge.properties:
                    obligations.append(
                        Obligation(
                            "READ",
                            Target(
                                "PROPERTIES",
                                self.graph_name,
                                edge_types=(
                                    frozenset(edge.required_labels)
                                    if edge.required_labels
                                    else frozenset()
                                ),
                                properties=frozenset({property_name}),
                            ),
                            f"{reason_prefix}: edge-pattern property {property_name}",
                        )
                    )

        for match in query.matches:
            add_pattern(match.pattern, "OPTIONAL MATCH" if match.optional else "MATCH")

        matched_nodes = set(variable_nodes)
        # New bindings contribute labels to RETURN checks, without requiring
        # TRAVERSE on resources that do not yet exist.
        for pattern in query.insert_patterns:
            for node in pattern.nodes:
                variable_nodes.setdefault(node.variable, set()).update(node.labels)
            for edge in pattern.edges:
                if edge.variable:
                    variable_edges.setdefault(edge.variable, set()).update(edge.required_labels)

        expressions = [query.where]
        expressions.extend(item.expression for item in query.return_items)
        expressions.extend(item.expression for item in query.order_by)
        expressions.extend(assignment.expression for assignment in query.set_assignments)

        def add_expression(
            item: Expr | None, node_scope: dict[str, set[str]], edge_scope: dict[str, set[str]]
        ) -> None:
            if isinstance(item, PropertyRef):
                target = Target(
                    "PROPERTIES",
                    self.graph_name,
                    labels=frozenset(node_scope.get(item.variable, set())),
                    edge_types=frozenset(edge_scope.get(item.variable, set())),
                    properties=frozenset({item.property_name}),
                )
                obligations.append(
                    Obligation("READ", target, f"dereference {item.variable}.{item.property_name}")
                )
            elif isinstance(item, ExistsExpr):
                # Local labels must not escape into a sibling or outer scope.
                nested_nodes = {name: set(labels) for name, labels in node_scope.items()}
                nested_edges = {name: set(labels) for name, labels in edge_scope.items()}
                for nested_match in item.query.matches:
                    add_pattern(nested_match.pattern, "EXISTS", nested_nodes, nested_edges)
                add_expression(item.query.where, nested_nodes, nested_edges)
            elif isinstance(item, BinaryExpr):
                add_expression(item.left, node_scope, edge_scope)
                add_expression(item.right, node_scope, edge_scope)
            elif isinstance(item, UnaryExpr):
                add_expression(item.operand, node_scope, edge_scope)
            elif isinstance(item, FunctionCall):
                for argument in item.args:
                    add_expression(argument, node_scope, edge_scope)

        for expression in expressions:
            add_expression(expression, variable_nodes, variable_edges)

        if query.update_kind == "SET":
            for set_assignment in query.set_assignments:
                labels = variable_nodes.get(set_assignment.variable, set())
                edge_types = variable_edges.get(set_assignment.variable, set())
                target = Target(
                    "PROPERTIES",
                    self.graph_name,
                    labels=frozenset(labels),
                    edge_types=frozenset(edge_types),
                    properties=frozenset({set_assignment.property_name}),
                )
                obligations.append(
                    Obligation(
                        "SET",
                        target,
                        f"SET {set_assignment.variable}.{set_assignment.property_name}",
                    )
                )
        elif query.update_kind == "REMOVE":
            for remove_assignment in query.remove_assignments:
                labels = variable_nodes.get(remove_assignment.variable, set())
                edge_types = variable_edges.get(remove_assignment.variable, set())
                target = Target(
                    "PROPERTIES",
                    self.graph_name,
                    labels=frozenset(labels),
                    edge_types=frozenset(edge_types),
                    properties=frozenset({remove_assignment.property_name}),
                )
                obligations.append(
                    Obligation(
                        "REMOVE",
                        target,
                        f"REMOVE {remove_assignment.variable}.{remove_assignment.property_name}",
                    )
                )
        elif query.update_kind == "DELETE":
            for variable in query.delete_variables:
                if variable in variable_nodes and variable_nodes[variable]:
                    target = Target(
                        "NODES",
                        self.graph_name,
                        labels=frozenset(variable_nodes[variable]),
                    )
                elif variable in variable_edges and variable_edges[variable]:
                    target = Target(
                        "EDGES",
                        self.graph_name,
                        edge_types=frozenset(variable_edges[variable]),
                    )
                else:
                    target = Target("GRAPH", self.graph_name)
                obligations.append(Obligation("DELETE", target, f"DELETE variable {variable}"))
        elif query.update_kind == "INSERT":
            if not query.insert_patterns:
                obligations.append(
                    Obligation("INSERT", Target("GRAPH", self.graph_name), "INSERT clause")
                )
            inserted_nodes = set()
            for pattern in query.insert_patterns:
                for node in pattern.nodes:
                    if node.variable in matched_nodes or node.variable in inserted_nodes:
                        continue
                    inserted_nodes.add(node.variable)
                    target = (
                        Target("NODES", self.graph_name, labels=frozenset(node.labels))
                        if node.labels
                        else Target("GRAPH", self.graph_name)
                    )
                    obligations.append(Obligation("INSERT", target, f"INSERT node {node.variable}"))
                for edge in pattern.edges:
                    target = (
                        Target(
                            "EDGES",
                            self.graph_name,
                            edge_types=frozenset(edge.required_labels),
                        )
                        if edge.required_labels
                        else Target("GRAPH", self.graph_name)
                    )
                    obligations.append(Obligation("INSERT", target, "INSERT edge"))

        deduplicated: dict[tuple[str, Target], Obligation] = {}
        for obligation in obligations:
            deduplicated[(obligation.action, obligation.target)] = obligation
        self.metrics.planning_ms += (perf_counter() - start) * 1000.0
        return list(deduplicated.values())

    def preflight(self, query: Query) -> list[Obligation]:
        obligations = self.derive_obligations(query)
        denied = [
            obligation
            for obligation in obligations
            if not self.catalog.object_target_permitted(
                self.user, obligation.action, obligation.target, conjunctive=True
            )
        ]
        if denied:
            rendered = "; ".join(
                f"{item.action} on {item.target.to_dict()} ({item.reason})" for item in denied
            )
            raise AuthorizationError(
                f"Static privilege obligations are not satisfied: {rendered}", code="42000"
            )
        return obligations

    def logical_plan(self, query: Query) -> dict[str, Any]:
        obligations = self.derive_obligations(query)
        operators: list[dict[str, Any]] = [
            {"operator": "GraphAccessCheck", "graph": self.graph_name, "user": self.user}
        ]
        for match in query.matches:
            operators.append(
                {
                    "operator": "OptionalGuardedMatch" if match.optional else "GuardedMatch",
                    "pattern": _pattern_to_dict(match.pattern),
                    "barrier": "check node/edge TRAVERSE before enqueue and binding materialization",
                }
            )
        if query.where is not None:
            operators.append(
                {
                    "operator": "SecureFilter",
                    "barrier": "READ checks precede protected property dereference",
                }
            )
        if query.order_by:
            operators.append({"operator": "SecureOrder", "items": len(query.order_by)})
        if query.update_kind:
            operators.append(
                {
                    "operator": "TransactionalWriteSet",
                    "kind": query.update_kind,
                    "detach": query.delete_detach if query.update_kind == "DELETE" else None,
                    "barrier": "USING on old state; WITH CHECK on prospective state; commit only after success",
                }
            )
        if query.return_items:
            operators.append(
                {
                    "operator": "SecureProjection",
                    "items": [item.alias for item in query.return_items],
                }
            )
        return {
            "user": self.user,
            "graph": self.graph_name,
            "match_mode": query.match_mode,
            "obligations": [item.to_dict() for item in obligations],
            "operators": operators,
            "optimizer_contract": [
                "authorization barriers cannot be crossed by property reads or user functions",
                "guard pushdown is allowed only when it preserves candidate-level decisions",
                "variable-length expansion must authorize every intermediate node and edge",
            ],
        }


def _pattern_to_dict(pattern: PatternChain) -> dict[str, Any]:
    return {
        "path_variable": pattern.path_variable,
        "path_mode": pattern.path_mode,
        "nodes": [
            {
                "variable": node.variable,
                "labels": sorted(node.labels),
                "properties": node.properties,
            }
            for node in pattern.nodes
        ],
        "edges": [
            {
                "variable": edge.variable,
                "type": edge.edge_type,
                "labels": sorted(edge.required_labels),
                "direction": edge.direction,
                "min_hops": edge.min_hops,
                "max_hops": edge.max_hops,
                "properties": edge.properties,
            }
            for edge in pattern.edges
        ],
    }
