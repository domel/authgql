from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator
import re

from .errors import ParseError


# ---------- Query AST ----------


@dataclass(frozen=True)
class ContextValue:
    name: str


@dataclass
class PatternNode:
    variable: str
    labels: set[str] = field(default_factory=set)
    properties: dict[str, Any] = field(default_factory=dict)
    anonymous: bool = False


@dataclass
class PatternEdge:
    variable: str | None = None
    edge_type: str | None = None
    direction: str = "out"
    min_hops: int = 1
    max_hops: int = 1
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class PatternChain:
    nodes: list[PatternNode]
    edges: list[PatternEdge]

    @property
    def variables(self) -> set[str]:
        result = {node.variable for node in self.nodes}
        result.update(edge.variable for edge in self.edges if edge.variable)
        return result


class Expr:
    pass


@dataclass
class Literal(Expr):
    value: Any


@dataclass
class VariableRef(Expr):
    name: str


@dataclass
class PropertyRef(Expr):
    variable: str
    property_name: str


@dataclass
class LabelTest(Expr):
    variable: str
    label: str


@dataclass
class UnaryExpr(Expr):
    op: str
    operand: Expr


@dataclass
class BinaryExpr(Expr):
    left: Expr
    op: str
    right: Expr


@dataclass
class FunctionCall(Expr):
    name: str
    args: list[Expr]


@dataclass
class ExistsExpr(Expr):
    query: "Query"


@dataclass
class MatchClause:
    pattern: PatternChain
    optional: bool = False


@dataclass
class ReturnItem:
    expression: Expr
    alias: str


@dataclass
class OrderItem:
    expression: Expr
    descending: bool = False


@dataclass
class SetAssignment:
    variable: str
    property_name: str
    expression: Expr


@dataclass
class RemoveAssignment:
    variable: str
    property_name: str


@dataclass
class InsertNode:
    variable: str
    labels: set[str]
    properties: dict[str, Any]


@dataclass
class Query:
    matches: list[MatchClause] = field(default_factory=list)
    where: Expr | None = None
    return_items: list[ReturnItem] = field(default_factory=list)
    order_by: list[OrderItem] = field(default_factory=list)
    limit: int | None = None
    update_kind: str | None = None
    set_assignments: list[SetAssignment] = field(default_factory=list)
    remove_assignments: list[RemoveAssignment] = field(default_factory=list)
    delete_variables: list[str] = field(default_factory=list)
    delete_detach: bool = False
    insert_nodes: list[InsertNode] = field(default_factory=list)
    insert_patterns: list[PatternChain] = field(default_factory=list)
    source: str = ""


# ---------- Generic splitting/scanning ----------


def strip_comments(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        quote: str | None = None
        escaped = False
        cut = len(line)
        for index, char in enumerate(line):
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if quote:
                if char == quote:
                    quote = None
                continue
            if char in {"'", '"'}:
                quote = char
                continue
            if char == "#" or (char == "-" and index + 1 < len(line) and line[index + 1] == "-"):
                cut = index
                break
        lines.append(line[:cut])
    return "\n".join(lines)


def split_top_level(text: str, delimiter: str = ",") -> list[str]:
    parts: list[str] = []
    start = 0
    paren = brace = bracket = 0
    quote: str | None = None
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char == "(":
            paren += 1
        elif char == ")":
            paren -= 1
        elif char == "{":
            brace += 1
        elif char == "}":
            brace -= 1
        elif char == "[":
            bracket += 1
        elif char == "]":
            bracket -= 1
        elif char == delimiter and paren == brace == bracket == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def _find_matching(text: str, start: int, opening: str, closing: str) -> int:
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return index
    raise ParseError(f"Unclosed {opening!r} in: {text}")


CLAUSE_KEYWORDS = [
    "OPTIONAL MATCH",
    "NODETACH DELETE",
    "DETACH DELETE",
    "ORDER BY",
    "MATCH",
    "WHERE",
    "RETURN",
    "SET",
    "REMOVE",
    "DELETE",
    "INSERT",
    "LIMIT",
    "FINISH",
]


def scan_clauses(text: str) -> list[tuple[str, str]]:
    upper = text.upper()
    positions: list[tuple[int, str]] = []
    paren = brace = bracket = 0
    quote: str | None = None
    escaped = False
    index = 0
    sorted_keywords = sorted(CLAUSE_KEYWORDS, key=len, reverse=True)
    while index < len(text):
        char = text[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if char == "\\":
            escaped = True
            index += 1
            continue
        if quote:
            if char == quote:
                quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            index += 1
            continue
        if char == "(":
            paren += 1
        elif char == ")":
            paren -= 1
        elif char == "{":
            brace += 1
        elif char == "}":
            brace -= 1
        elif char == "[":
            bracket += 1
        elif char == "]":
            bracket -= 1
        if paren == brace == bracket == 0:
            matched = None
            for keyword in sorted_keywords:
                if upper.startswith(keyword, index):
                    before_ok = index == 0 or not (upper[index - 1].isalnum() or upper[index - 1] == "_")
                    end = index + len(keyword)
                    after_ok = end == len(text) or not (upper[end].isalnum() or upper[end] == "_")
                    if before_ok and after_ok:
                        matched = keyword
                        break
            if matched:
                positions.append((index, matched))
                index += len(matched)
                continue
        index += 1

    if not positions:
        raise ParseError("No supported GQL clause was found")
    prefix = text[: positions[0][0]].strip()
    if prefix:
        raise ParseError(f"Unexpected input before first clause: {prefix!r}")
    result: list[tuple[str, str]] = []
    for pos_index, (position, keyword) in enumerate(positions):
        content_start = position + len(keyword)
        content_end = positions[pos_index + 1][0] if pos_index + 1 < len(positions) else len(text)
        result.append((keyword, text[content_start:content_end].strip()))
    return result


# ---------- Literals and patterns ----------


_NUMBER_RE = re.compile(r"^[+-]?(?:\d+\.\d+|\d+)$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_literal(text: str) -> Any:
    text = text.strip()
    if not text:
        raise ParseError("Empty literal")
    if (text[0] == text[-1]) and text[0] in {"'", '"'}:
        value = text[1:-1]
        return bytes(value, "utf-8").decode("unicode_escape")
    upper = text.upper()
    if upper == "TRUE":
        return True
    if upper == "FALSE":
        return False
    if upper == "NULL":
        return None
    if upper == "SESSION_USER":
        return ContextValue(upper)
    if _NUMBER_RE.match(text):
        return float(text) if "." in text else int(text)
    raise ParseError(f"Unsupported literal: {text!r}")


def parse_property_map(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        return {}
    if not (text.startswith("{") and text.endswith("}")):
        raise ParseError(f"Expected a property map, got {text!r}")
    body = text[1:-1].strip()
    if not body:
        return {}
    result: dict[str, Any] = {}
    for item in split_top_level(body):
        if ":" not in item:
            raise ParseError(f"Invalid property map entry: {item!r}")
        key, value = item.split(":", 1)
        key = key.strip().strip("`\"")
        if not _IDENTIFIER_RE.match(key):
            raise ParseError(f"Invalid property name: {key!r}")
        result[key] = parse_literal(value)
    return result


class PatternParser:
    def __init__(self) -> None:
        self.anonymous_counter = 0

    def parse_chain(self, text: str) -> PatternChain:
        text = text.strip()
        position = 0
        nodes: list[PatternNode] = []
        edges: list[PatternEdge] = []
        node, position = self._parse_node_at(text, position)
        nodes.append(node)
        while True:
            position = self._skip_ws(text, position)
            if position >= len(text):
                break
            edge, position = self._parse_edge_at(text, position)
            edges.append(edge)
            node, position = self._parse_node_at(text, position)
            nodes.append(node)
        if len(nodes) != len(edges) + 1:
            raise ParseError(f"Malformed path pattern: {text!r}")
        return PatternChain(nodes=nodes, edges=edges)

    @staticmethod
    def _skip_ws(text: str, position: int) -> int:
        while position < len(text) and text[position].isspace():
            position += 1
        return position

    def _parse_node_at(self, text: str, position: int) -> tuple[PatternNode, int]:
        position = self._skip_ws(text, position)
        if position >= len(text) or text[position] != "(":
            raise ParseError(f"Expected node pattern at: {text[position:]!r}")
        end = _find_matching(text, position, "(", ")")
        token = text[position + 1 : end].strip()
        return self._parse_node_token(token), end + 1

    def _parse_node_token(self, token: str) -> PatternNode:
        property_map: dict[str, Any] = {}
        brace_index = self._top_level_char(token, "{")
        if brace_index is not None:
            brace_end = _find_matching(token, brace_index, "{", "}")
            if token[brace_end + 1 :].strip():
                raise ParseError(f"Unexpected text after node property map: {token!r}")
            property_map = parse_property_map(token[brace_index : brace_end + 1])
            token = token[:brace_index].strip()

        variable_match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)", token)
        if variable_match:
            variable = variable_match.group(1)
            anonymous = False
        else:
            self.anonymous_counter += 1
            variable = f"_anon{self.anonymous_counter}"
            anonymous = True
        labels = set(re.findall(r":\s*([A-Za-z_][A-Za-z0-9_]*)", token))
        return PatternNode(variable, labels, property_map, anonymous)

    def _parse_edge_at(self, text: str, position: int) -> tuple[PatternEdge, int]:
        position = self._skip_ws(text, position)
        direction: str
        if text.startswith("<-[", position):
            direction = "in"
            open_bracket = position + 2
            close_bracket = _find_matching(text, open_bracket, "[", "]")
            if not text.startswith("-", close_bracket + 1):
                raise ParseError("Inbound edge pattern must end with ]-")
            next_position = close_bracket + 2
        elif text.startswith("-[", position):
            open_bracket = position + 1
            close_bracket = _find_matching(text, open_bracket, "[", "]")
            if text.startswith("->", close_bracket + 1):
                direction = "out"
                next_position = close_bracket + 3
            elif text.startswith("-", close_bracket + 1):
                direction = "both"
                next_position = close_bracket + 2
            else:
                raise ParseError("Edge pattern must end with ]-> or ]-")
        elif text.startswith("~[", position):
            direction = "both"
            open_bracket = position + 1
            close_bracket = _find_matching(text, open_bracket, "[", "]")
            if not text.startswith("~", close_bracket + 1):
                raise ParseError("Undirected edge pattern must end with ]~")
            next_position = close_bracket + 2
        else:
            raise ParseError(f"Expected edge pattern at: {text[position:]!r}")
        token = text[open_bracket + 1 : close_bracket].strip()
        return self._parse_edge_token(token, direction), next_position

    def _parse_edge_token(self, token: str, direction: str) -> PatternEdge:
        properties: dict[str, Any] = {}
        brace_index = self._top_level_char(token, "{")
        if brace_index is not None:
            brace_end = _find_matching(token, brace_index, "{", "}")
            properties = parse_property_map(token[brace_index : brace_end + 1])
            token = (token[:brace_index] + token[brace_end + 1 :]).strip()

        min_hops = max_hops = 1
        quantifier = re.search(r"\*(?:(\d*)\.\.(\d*)|(\d+))?", token)
        if quantifier:
            if quantifier.group(3):
                min_hops = max_hops = int(quantifier.group(3))
            elif quantifier.group(1) is not None or quantifier.group(2) is not None:
                min_hops = int(quantifier.group(1) or 0)
                max_hops = int(quantifier.group(2) or 5)
            else:
                min_hops, max_hops = 1, 5
            if max_hops > 12:
                raise ParseError("Prototype limits variable-length paths to at most 12 hops")
            token = token[: quantifier.start()] + token[quantifier.end() :]

        token = token.strip()
        variable: str | None = None
        edge_type: str | None = None
        if ":" in token:
            before, after = token.split(":", 1)
            before = before.strip()
            after = after.strip()
            variable = before or None
            edge_type_match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", after)
            if not edge_type_match:
                raise ParseError(f"Invalid edge-label shorthand in {token!r}")
            edge_type = edge_type_match.group(1)
        elif token:
            if not _IDENTIFIER_RE.match(token):
                raise ParseError(f"Invalid edge variable: {token!r}")
            variable = token
        return PatternEdge(variable, edge_type, direction, min_hops, max_hops, properties)

    @staticmethod
    def _top_level_char(text: str, wanted: str) -> int | None:
        quote: str | None = None
        escaped = False
        depth = 0
        for index, char in enumerate(text):
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if quote:
                if char == quote:
                    quote = None
                continue
            if char in {"'", '"'}:
                quote = char
                continue
            if char == wanted and depth == 0:
                return index
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
        return None


# ---------- Expression parser ----------


@dataclass(frozen=True)
class Token:
    kind: str
    value: str


_TOKEN_RE = re.compile(
    r"\s*(?:"
    r"(?P<STRING>'(?:\\.|[^'])*'|\"(?:\\.|[^\"])*\")|"
    r"(?P<NUMBER>[+-]?(?:\d+\.\d+|\d+))|"
    r"(?P<OP><>|!=|<=|>=|=|<|>)|"
    r"(?P<LPAREN>\()|(?P<RPAREN>\))|"
    r"(?P<DOT>\.)|(?P<COLON>:)|(?P<COMMA>,)|(?P<STAR>\*)|"
    r"(?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)"
    r")"
)


def _extract_exists(text: str) -> tuple[str, dict[str, str]]:
    replacements: dict[str, str] = {}
    upper = text.upper()
    output: list[str] = []
    index = 0
    quote: str | None = None
    escaped = False
    while index < len(text):
        char = text[index]
        if escaped:
            output.append(char)
            escaped = False
            index += 1
            continue
        if char == "\\":
            output.append(char)
            escaped = True
            index += 1
            continue
        if quote:
            output.append(char)
            if char == quote:
                quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            output.append(char)
            quote = char
            index += 1
            continue
        if upper.startswith("EXISTS", index):
            before_ok = index == 0 or not (upper[index - 1].isalnum() or upper[index - 1] == "_")
            end_word = index + 6
            after_ok = end_word == len(text) or not (upper[end_word].isalnum() or upper[end_word] == "_")
            if before_ok and after_ok:
                brace_start = end_word
                while brace_start < len(text) and text[brace_start].isspace():
                    brace_start += 1
                if brace_start < len(text) and text[brace_start] == "{":
                    brace_end = _find_matching(text, brace_start, "{", "}")
                    placeholder = f"__EXISTS_{len(replacements)}__"
                    replacements[placeholder] = text[brace_start + 1 : brace_end]
                    output.append(placeholder)
                    index = brace_end + 1
                    continue
        output.append(char)
        index += 1
    return "".join(output), replacements


def tokenize_expression(text: str) -> list[Token]:
    normalized, exists_blocks = _extract_exists(text)
    tokens: list[Token] = []
    position = 0
    while position < len(normalized):
        match = _TOKEN_RE.match(normalized, position)
        if not match:
            if normalized[position:].strip() == "":
                break
            raise ParseError(f"Cannot tokenize expression near: {normalized[position:]!r}")
        kind = match.lastgroup
        value = match.group(kind) if kind else ""
        if kind == "IDENT" and value in exists_blocks:
            tokens.append(Token("EXISTS_BLOCK", exists_blocks[value]))
        else:
            tokens.append(Token(kind or "", value))
        position = match.end()
    tokens.append(Token("EOF", ""))
    return tokens


class ExpressionParser:
    def __init__(self, text: str) -> None:
        self.tokens = tokenize_expression(text)
        self.position = 0

    def current(self) -> Token:
        return self.tokens[self.position]

    def advance(self) -> Token:
        token = self.current()
        self.position += 1
        return token

    def accept(self, kind: str, value: str | None = None) -> Token | None:
        token = self.current()
        if token.kind != kind:
            return None
        if value is not None and token.value.upper() != value.upper():
            return None
        self.position += 1
        return token

    def expect(self, kind: str, value: str | None = None) -> Token:
        token = self.accept(kind, value)
        if token is None:
            wanted = f"{kind} {value}" if value else kind
            raise ParseError(f"Expected {wanted}, got {self.current()}")
        return token

    def parse(self) -> Expr:
        result = self.parse_or()
        self.expect("EOF")
        return result

    def parse_or(self) -> Expr:
        result = self.parse_and()
        while self.accept("IDENT", "OR"):
            result = BinaryExpr(result, "OR", self.parse_and())
        return result

    def parse_and(self) -> Expr:
        result = self.parse_not()
        while self.accept("IDENT", "AND"):
            result = BinaryExpr(result, "AND", self.parse_not())
        return result

    def parse_not(self) -> Expr:
        if self.accept("IDENT", "NOT"):
            return UnaryExpr("NOT", self.parse_not())
        return self.parse_comparison()

    def parse_comparison(self) -> Expr:
        left = self.parse_primary()
        if self.current().kind == "OP":
            op = self.advance().value
            return BinaryExpr(left, op, self.parse_primary())
        if self.accept("IDENT", "IS"):
            negate = bool(self.accept("IDENT", "NOT"))
            self.expect("IDENT", "NULL")
            return UnaryExpr("IS NOT NULL" if negate else "IS NULL", left)
        if self.accept("IDENT", "IN"):
            self.expect("LPAREN")
            values: list[Expr] = []
            if not self.accept("RPAREN"):
                while True:
                    values.append(self.parse_or())
                    if self.accept("RPAREN"):
                        break
                    self.expect("COMMA")
            return BinaryExpr(left, "IN", FunctionCall("LIST", values))
        return left

    def parse_primary(self) -> Expr:
        token = self.current()
        if self.accept("LPAREN"):
            result = self.parse_or()
            self.expect("RPAREN")
            return result
        if token.kind == "STRING":
            self.advance()
            return Literal(parse_literal(token.value))
        if token.kind == "NUMBER":
            self.advance()
            return Literal(parse_literal(token.value))
        if token.kind == "EXISTS_BLOCK":
            self.advance()
            return ExistsExpr(parse_query(token.value, allow_no_return=True))
        if token.kind == "STAR":
            self.advance()
            return VariableRef("*")
        if token.kind == "IDENT":
            name = self.advance().value
            upper = name.upper()
            if upper in {"TRUE", "FALSE", "NULL"}:
                return Literal(parse_literal(upper))
            if upper == "SESSION_USER":
                return VariableRef(upper)
            if self.accept("LPAREN"):
                args: list[Expr] = []
                if not self.accept("RPAREN"):
                    while True:
                        args.append(self.parse_or())
                        if self.accept("RPAREN"):
                            break
                        self.expect("COMMA")
                return FunctionCall(upper, args)
            if self.accept("DOT"):
                prop = self.expect("IDENT").value
                return PropertyRef(name, prop)
            if self.accept("COLON"):
                label = self.expect("IDENT").value
                return LabelTest(name, label)
            return VariableRef(name)
        raise ParseError(f"Unexpected token in expression: {token}")


def parse_expression(text: str) -> Expr:
    return ExpressionParser(text).parse()


# ---------- Whole query parser ----------


def _parse_return_items(text: str) -> list[ReturnItem]:
    result: list[ReturnItem] = []
    for item in split_top_level(text):
        alias_match = re.match(r"^(.*?)(?:\s+AS\s+([A-Za-z_][A-Za-z0-9_]*))$", item, re.I | re.S)
        if alias_match:
            expression_text = alias_match.group(1).strip()
            alias = alias_match.group(2)
        else:
            expression_text = item.strip()
            alias = re.sub(r"[^A-Za-z0-9_]+", "_", expression_text).strip("_") or "value"
        result.append(ReturnItem(parse_expression(expression_text), alias))
    return result


def _parse_order_items(text: str) -> list[OrderItem]:
    result: list[OrderItem] = []
    for item in split_top_level(text):
        match = re.match(r"^(.*?)(?:\s+(ASC|DESC))?$", item.strip(), re.I | re.S)
        if not match:
            raise ParseError(f"Invalid ORDER BY item: {item!r}")
        result.append(OrderItem(parse_expression(match.group(1).strip()), (match.group(2) or "").upper() == "DESC"))
    return result


def _parse_set_assignments(text: str) -> list[SetAssignment]:
    result: list[SetAssignment] = []
    for item in split_top_level(text):
        match = re.match(
            r"^([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$",
            item,
            re.S,
        )
        if not match:
            raise ParseError(f"Invalid SET assignment: {item!r}")
        result.append(SetAssignment(match.group(1), match.group(2), parse_expression(match.group(3))))
    return result


def _parse_remove_assignments(text: str) -> list[RemoveAssignment]:
    result: list[RemoveAssignment] = []
    for item in split_top_level(text):
        match = re.fullmatch(
            r"([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)",
            item.strip(),
        )
        if not match:
            raise ParseError(
                "The prototype supports property removal as REMOVE variable.property"
            )
        result.append(RemoveAssignment(match.group(1), match.group(2)))
    return result


def parse_query(text: str, allow_no_return: bool = False) -> Query:
    source = strip_comments(text).strip().rstrip(";").strip()
    if not source:
        raise ParseError("Empty GQL program")
    clauses = scan_clauses(source)
    pattern_parser = PatternParser()
    query = Query(source=source)
    seen_update = False

    for keyword, content in clauses:
        if keyword in {"MATCH", "OPTIONAL MATCH"}:
            if seen_update:
                raise ParseError("MATCH cannot follow a data-modifying clause in this prototype")
            for chain_text in split_top_level(content):
                query.matches.append(
                    MatchClause(pattern_parser.parse_chain(chain_text), optional=keyword == "OPTIONAL MATCH")
                )
        elif keyword == "WHERE":
            if query.where is not None:
                raise ParseError("Only one top-level WHERE clause is supported")
            query.where = parse_expression(content)
        elif keyword == "RETURN":
            query.return_items = _parse_return_items(content)
        elif keyword == "ORDER BY":
            query.order_by = _parse_order_items(content)
        elif keyword == "LIMIT":
            try:
                query.limit = int(content)
            except ValueError as exc:
                raise ParseError("LIMIT must be an integer") from exc
            if query.limit < 0:
                raise ParseError("LIMIT must be non-negative")
        elif keyword == "SET":
            if query.update_kind is not None:
                raise ParseError("Only one update clause is supported")
            query.update_kind = "SET"
            query.set_assignments = _parse_set_assignments(content)
            seen_update = True
        elif keyword == "REMOVE":
            if query.update_kind is not None:
                raise ParseError("Only one update clause is supported")
            query.update_kind = "REMOVE"
            query.remove_assignments = _parse_remove_assignments(content)
            seen_update = True
        elif keyword in {"DELETE", "DETACH DELETE", "NODETACH DELETE"}:
            if query.update_kind is not None:
                raise ParseError("Only one update clause is supported")
            variables = [item.strip() for item in split_top_level(content)]
            if not variables or not all(_IDENTIFIER_RE.match(item) for item in variables):
                raise ParseError("DELETE expects one or more variables")
            query.update_kind = "DELETE"
            query.delete_variables = variables
            query.delete_detach = keyword == "DETACH DELETE"
            seen_update = True
        elif keyword == "INSERT":
            if query.update_kind is not None:
                raise ParseError("Only one update clause is supported")
            query.update_kind = "INSERT"
            for item in split_top_level(content):
                chain = pattern_parser.parse_chain(item)
                if any(edge.min_hops != 1 or edge.max_hops != 1 for edge in chain.edges):
                    raise ParseError("INSERT edge patterns cannot be quantified")
                query.insert_patterns.append(chain)
                if not chain.edges and len(chain.nodes) == 1:
                    node = chain.nodes[0]
                    query.insert_nodes.append(InsertNode(node.variable, node.labels, node.properties))
            seen_update = True
        elif keyword == "FINISH":
            query.return_items = []
        else:
            raise ParseError(f"Unsupported clause: {keyword}")

    if not query.matches and query.update_kind != "INSERT":
        raise ParseError("A query must contain MATCH or INSERT")
    if not allow_no_return and not query.return_items and query.update_kind is None:
        raise ParseError("Read-only queries require RETURN")
    return query


def walk_expr(expr: Expr | None) -> Iterator[Expr]:
    if expr is None:
        return
    yield expr
    if isinstance(expr, UnaryExpr):
        yield from walk_expr(expr.operand)
    elif isinstance(expr, BinaryExpr):
        yield from walk_expr(expr.left)
        yield from walk_expr(expr.right)
    elif isinstance(expr, FunctionCall):
        for arg in expr.args:
            yield from walk_expr(arg)
    elif isinstance(expr, ExistsExpr):
        if expr.query.where:
            yield from walk_expr(expr.query.where)
        for item in expr.query.return_items:
            yield from walk_expr(item.expression)


def property_references(query: Query) -> list[PropertyRef]:
    expressions: list[Expr | None] = [query.where]
    expressions.extend(item.expression for item in query.return_items)
    expressions.extend(item.expression for item in query.order_by)
    expressions.extend(assignment.expression for assignment in query.set_assignments)
    result: list[PropertyRef] = []
    for expression in expressions:
        for subexpression in walk_expr(expression):
            if isinstance(subexpression, PropertyRef):
                result.append(subexpression)
    return result
