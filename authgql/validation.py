"""Static name/type/arity checks for the supported query and policy ASTs."""

from typing import Any
from .errors import ParseError
from .parser import (
    Literal,
    VariableRef,
    PropertyRef,
    LabelTest,
    UnaryExpr,
    BinaryExpr,
    FunctionCall,
    ExistsExpr,
    ContextValue,
    ReturnItem,
    Expr,
    Query,
    PatternChain,
    PatternNode,
    PatternEdge,
    SetAssignment,
    RemoveAssignment,
    order_expression,
)
from .values import kind, parameter


def require(actual: str, allowed: set[str]) -> None:
    if actual not in {*allowed, "ANY", "NULL"}:
        raise ParseError("Expression has an incompatible type for this context")


def expression_type(
    expr: Expr,
    scope: dict[str, str],
    *,
    policy: bool = False,
    parameters: dict[str, Any] | None = None,
    aggregate: bool = False,
) -> str:
    def visit(child: Expr) -> str:
        return expression_type(child, scope, policy=policy, parameters=parameters)

    if isinstance(expr, Literal):
        return kind(expr.value)
    if isinstance(expr, VariableRef):
        if expr.name == "SESSION_USER":
            return "STRING"
        if expr.name.startswith("$"):
            return "ANY" if parameters is None else kind(parameter(parameters, expr.name[1:]))
        if expr.name not in scope:
            raise ParseError(f"Unbound variable: {expr.name}")
        return scope[expr.name]
    if isinstance(expr, (PropertyRef, LabelTest)):
        if expr.variable not in scope:
            raise ParseError(f"Unbound element variable: {expr.variable}")
        require(scope[expr.variable], {"NODE", "EDGE", "ELEMENT"})
        return "BOOL" if isinstance(expr, LabelTest) else "ANY"
    if isinstance(expr, UnaryExpr):
        operand = visit(expr.operand)
        if expr.op == "NOT":
            require(operand, {"BOOL"})
        elif expr.op not in {"IS NULL", "IS NOT NULL"}:
            raise ParseError("Unsupported unary operation")
        return "BOOL"
    if isinstance(expr, BinaryExpr):
        left, right = visit(expr.left), visit(expr.right)
        if expr.op in {"AND", "OR"}:
            require(left, {"BOOL"})
            require(right, {"BOOL"})
        elif expr.op == "IN":
            require(right, {"LIST"})
        elif expr.op in {"=", "==", "!=", "<>", "<", "<=", ">", ">="}:
            if expr.op in {"<", "<=", ">", ">="}:
                require(left, {"NUMBER", "STRING"})
                require(right, {"NUMBER", "STRING"})
            element_types = {"NODE", "EDGE", "ELEMENT"}
            if left not in {"ANY", "NULL"} and right not in {"ANY", "NULL"}:
                if left != right and not {left, right}.issubset(element_types):
                    raise ParseError("Incompatible comparison types")
        else:
            raise ParseError("Unsupported binary operation")
        return "BOOL"
    if isinstance(expr, FunctionCall):
        name = expr.name.upper()
        arities = {
            "ID": (1, 1),
            "ELEMENT_ID": (1, 1),
            "COALESCE": (1, None),
            "LIST": (0, None),
            "PARAM": (1, 1),
            "COUNT": (1, 1),
        }
        if policy:
            arities.update({"HAS_ROLE": (1, 1), "PROPERTY_EXISTS": (2, 2)})
        if name not in arities:
            raise ParseError("Unsupported function")
        minimum, maximum = arities[name]
        if len(expr.args) < minimum or (maximum is not None and len(expr.args) > maximum):
            raise ParseError(f"Incorrect arity for {name}")
        if name == "COUNT":
            if policy or not aggregate:
                raise ParseError("COUNT is allowed only as a top-level RETURN aggregate")
            arg = expr.args[0]
            if not (isinstance(arg, VariableRef) and arg.name == "*"):
                visit(arg)
            return "NUMBER"
        types = [visit(arg) for arg in expr.args]
        if name in {"ID", "ELEMENT_ID"}:
            require(types[0], {"NODE", "EDGE", "ELEMENT"})
            return "STRING"
        if name == "PARAM":
            require(types[0], {"STRING"})
            if not isinstance(expr.args[0], Literal) or not isinstance(expr.args[0].value, str):
                raise ParseError("PARAM requires a literal parameter name in this fragment")
            return "ANY" if parameters is None else kind(parameter(parameters, expr.args[0].value))
        if name == "HAS_ROLE":
            require(types[0], {"STRING"})
            return "BOOL"
        if name == "PROPERTY_EXISTS":
            require(types[0], {"NODE", "EDGE", "ELEMENT"})
            require(types[1], {"STRING"})
            return "BOOL"
        if name == "LIST":
            return "LIST"
        concrete = set(types) - {"NULL", "ANY"}
        if len(concrete) > 1:
            raise ParseError("COALESCE arguments must have compatible types")
        return "ANY" if "ANY" in types else next(iter(concrete), "NULL")
    if isinstance(expr, ExistsExpr):
        if expr.query.update_kind:
            raise ParseError("EXISTS cannot contain data modification")
        if expr.query.return_items or expr.query.order_by or expr.query.limit is not None:
            raise ParseError("This fragment's EXISTS accepts MATCH and WHERE only")
        validate_query(expr.query, outer=scope, policy=policy, parameters=parameters)
        return "BOOL"
    raise ParseError("Unsupported expression AST")


def validate_query(
    query: Query,
    *,
    outer: dict[str, str] | None = None,
    policy: bool = False,
    parameters: dict[str, Any] | None = None,
) -> dict[str, str]:
    scope = dict(outer or {})
    if query.match_mode != "REPEATABLE ELEMENTS":
        raise ParseError("Unsupported match mode")

    def bind(name: str, category: str) -> None:
        if name in scope and scope[name] not in {category, "ELEMENT"}:
            raise ParseError(f"Conflicting variable kind: {name}")
        scope[name] = category

    def pattern(chain: PatternChain) -> None:
        if chain.path_mode != "WALK":
            raise ParseError("Unsupported path mode")
        for node in chain.nodes:
            bind(node.variable, "NODE")
        for edge in chain.edges:
            if not 0 <= edge.min_hops <= edge.max_hops <= 12:
                raise ParseError("Invalid path bounds")
            if edge.variable:
                bind(edge.variable, "LIST" if edge.quantified else "EDGE")
        if chain.path_variable:
            if chain.path_variable in {n.variable for n in chain.nodes} | {
                e.variable for e in chain.edges
            }:
                raise ParseError("Path and element variables must differ")
            bind(chain.path_variable, "PATH")
        parts: list[PatternNode | PatternEdge] = [*chain.nodes, *chain.edges]
        for part in parts:
            for value in part.properties.values():
                if (
                    isinstance(value, ContextValue)
                    and value.name.startswith("$")
                    and parameters is not None
                ):
                    parameter(parameters, value.name[1:])

    for clause in query.matches:
        pattern(clause.pattern)
    if query.where is not None:
        require(expression_type(query.where, scope, policy=policy, parameters=parameters), {"BOOL"})
    targets: list[SetAssignment | RemoveAssignment] = [
        *query.set_assignments,
        *query.remove_assignments,
    ]
    for target in targets:
        if target.variable not in scope:
            raise ParseError("Unbound update variable")
        require(scope[target.variable], {"NODE", "EDGE", "ELEMENT"})
    set_keys = [(item.variable, item.property_name) for item in query.set_assignments]
    if len(set_keys) != len(set(set_keys)):
        raise ParseError("SET cannot repeat the same variable/property target")
    for assignment in query.set_assignments:
        category = expression_type(assignment.expression, scope, parameters=parameters)
        require(category, {"BOOL", "NUMBER", "STRING", "LIST"})
    for name in query.delete_variables:
        if name not in scope:
            raise ParseError("Unbound DELETE variable")
        require(scope[name], {"NODE", "EDGE", "ELEMENT"})
    # INSERT has declaration rules distinct from MATCH's variable reuse.
    edge_names = [
        edge.variable for chain in query.insert_patterns for edge in chain.edges if edge.variable
    ]
    node_names = {node.variable for chain in query.insert_patterns for node in chain.nodes}
    if len(edge_names) != len(set(edge_names)) or set(edge_names).intersection(
        set(scope) | node_names
    ):
        raise ParseError("An INSERT edge variable must be new and declared exactly once")
    for chain in query.insert_patterns:
        if chain.path_variable or any(edge.quantified for edge in chain.edges):
            raise ParseError("INSERT does not support path bindings or quantifiers")
        if any(edge.direction not in {"in", "out", "undirected"} for edge in chain.edges):
            raise ParseError("INSERT requires a directed or undirected edge pattern")
        for node in chain.nodes:
            if node.variable in scope:
                require(scope[node.variable], {"NODE"})
                if node.decorated or node.labels or node.properties:
                    raise ParseError(
                        "A bound or repeated INSERT node cannot specify labels or properties"
                    )
            bind(node.variable, "NODE")
        pattern(chain)
    if query.return_all:
        query.return_items = [
            ReturnItem(VariableRef(name), name)
            for name in sorted(scope)
            if not name.startswith("@")
        ]
        if not query.return_items:
            raise ParseError("RETURN * requires at least one named binding")
    aliases = [item.alias for item in query.return_items]
    if len(aliases) != len(set(aliases)):
        raise ParseError("Duplicate RETURN alias")
    aggregates = [
        isinstance(item.expression, FunctionCall) and item.expression.name.upper() == "COUNT"
        for item in query.return_items
    ]
    if any(aggregates) and not all(aggregates):
        raise ParseError("Mixed grouping is outside this fragment")
    for item in query.return_items:
        expression_type(
            item.expression, scope, policy=policy, parameters=parameters, aggregate=True
        )
    for order_item in query.order_by:
        ordered = order_expression(query, order_item.expression)
        if any(aggregates) and not (
            isinstance(ordered, Literal)
            or isinstance(ordered, FunctionCall)
            and ordered.name.upper() == "COUNT"
        ):
            raise ParseError(
                "Aggregate ordering requires a COUNT expression, its alias or a constant"
            )
        category = expression_type(
            ordered,
            scope,
            policy=policy,
            parameters=parameters,
            aggregate=bool(aggregates) and all(aggregates),
        )
        require(category, {"NUMBER", "STRING", "BOOL"})
    return scope
