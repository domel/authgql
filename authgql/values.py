"""Typed scalar operations for the executable fragment (not Python coercion)."""

import math
from typing import Any
from .errors import ExecutionError
from .model import Node, Edge, PathValue, element_key


def kind(value: Any) -> str:
    if value is None:
        return "NULL"
    if type(value) is bool:
        return "BOOL"
    if type(value) in {int, float}:
        if isinstance(value, float) and not math.isfinite(value):
            raise ExecutionError("Non-finite numeric value", code="22000")
        return "NUMBER"
    if isinstance(value, str):
        return "STRING"
    if isinstance(value, (Node, Edge)):
        return "ELEMENT"
    if isinstance(value, PathValue):
        return "PATH"
    if isinstance(value, (list, tuple)):
        return "LIST"
    raise ExecutionError("Unsupported value type in this fragment", code="22000")


def truth(value: Any) -> bool | None:
    if value is None or type(value) is bool:
        return value
    raise ExecutionError("Boolean value required", code="22000")


def compare(left: Any, op: str, right: Any) -> bool | None:
    op = op.upper()
    if op == "IN":
        if right is None:
            return None
        if not isinstance(right, (list, tuple)):
            raise ExecutionError("IN requires a list", code="22000")
        outcomes = [compare(left, "=", item) for item in right]
        return True if True in outcomes else (None if None in outcomes else False)
    if left is None or right is None:
        return None
    if kind(left) != kind(right):
        raise ExecutionError("Incompatible comparison types", code="22000")
    category = kind(left)
    equal: bool | None
    if op in {"=", "==", "!=", "<>"}:
        if category == "ELEMENT":
            equal = element_key(left) == element_key(right)
        elif category == "LIST":
            if len(left) != len(right):
                equal = False
            else:
                outcomes = [compare(a, "=", b) for a, b in zip(left, right)]
                equal = False if False in outcomes else (None if None in outcomes else True)
        else:
            equal = left == right
        return equal if op in {"=", "=="} or equal is None else not equal
    if category not in {"NUMBER", "STRING"}:
        raise ExecutionError("Ordered comparison requires numeric or string operands", code="22000")
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    if op == ">":
        return left > right
    if op == ">=":
        return left >= right
    raise ExecutionError("Unsupported comparison operator", code="22000")


def scalar_value(value: Any) -> Any:
    """Validate stored/parameter values, including nested list contents."""
    category = kind(value)
    if category == "LIST":
        for child in value:
            scalar_value(child)
    elif category not in {"NULL", "BOOL", "NUMBER", "STRING"}:
        raise ExecutionError(
            "Properties and parameters cannot contain graph bindings", code="22000"
        )
    return value


def parameter(parameters: dict[str, Any], name: str) -> Any:
    if name not in parameters:
        raise ExecutionError(f"Unbound parameter: {name}", code="22000")
    return scalar_value(parameters[name])
