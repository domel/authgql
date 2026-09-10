from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator
import copy
import json

from .errors import ExecutionError


@dataclass
class Node:
    id: str
    labels: set[str] = field(default_factory=set)
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return "NODE"

    def clone(self) -> "Node":
        return Node(self.id, set(self.labels), copy.deepcopy(self.properties))


@dataclass(init=False)
class Edge:
    id: str
    type: str
    source: str
    target: str
    properties: dict[str, Any] = field(default_factory=dict)
    labels: set[str] = field(default_factory=set)
    directed: bool = True

    def __init__(
        self,
        id: str,
        type: str,
        source: str,
        target: str,
        properties: dict[str, Any] | None = None,
        labels: set[str] | None = None,
        directed: bool = True,
    ) -> None:
        self.id, self.type, self.source, self.target = id, type, source, target
        self.properties = {} if properties is None else properties
        # `type` remains a legacy input alias, never the authorization label set.
        self.labels = {self.type} if labels is None else set(labels)
        self.directed = directed
        self.__post_init__()

    def __post_init__(self) -> None:
        if type(self.directed) is not bool:
            raise ExecutionError("Edge directed flag must be Boolean", code="22000")

    @property
    def kind(self) -> str:
        return "EDGE"

    def clone(self) -> "Edge":
        return Edge(
            self.id,
            self.type,
            self.source,
            self.target,
            copy.deepcopy(self.properties),
            set(self.labels),
            self.directed,
        )


GraphElement = Node | Edge


def element_key(element: GraphElement) -> tuple[str, str]:
    return element.kind, element.id


@dataclass(frozen=True)
class PathValue:
    """An admitted path contains identities only, never property maps."""

    nodes: tuple[str, ...]
    edges: tuple[str, ...]


@dataclass
class PropertyGraph:
    name: str
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[str, Edge] = field(default_factory=dict)
    _out: dict[str, list[str]] = field(default_factory=dict, init=False, repr=False)
    _in: dict[str, list[str]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.rebuild_indexes()

    def rebuild_indexes(self) -> None:
        self._out = {node_id: [] for node_id in self.nodes}
        self._in = {node_id: [] for node_id in self.nodes}
        for edge in self.edges.values():
            if edge.source not in self.nodes or edge.target not in self.nodes:
                raise ExecutionError(
                    f"Edge {edge.id!r} references a missing endpoint", code="G2000"
                )
            self._out.setdefault(edge.source, []).append(edge.id)
            self._in.setdefault(edge.target, []).append(edge.id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PropertyGraph":
        nodes: dict[str, Node] = {}
        for raw in data.get("nodes", []):
            node = Node(
                id=str(raw["id"]),
                labels=set(map(str, raw.get("labels", []))),
                properties=copy.deepcopy(raw.get("properties", {})),
            )
            if node.id in nodes:
                raise ExecutionError(f"Duplicate node id: {node.id}", code="23000")
            nodes[node.id] = node

        edges: dict[str, Edge] = {}
        for raw in data.get("edges", []):
            edge = Edge(
                id=str(raw["id"]),
                type=str(raw.get("type", next(iter(raw.get("labels", [])), ""))),
                source=str(raw["source"]),
                target=str(raw["target"]),
                properties=copy.deepcopy(raw.get("properties", {})),
                labels=set(map(str, raw["labels"])) if "labels" in raw else None,
                directed=raw.get("directed", True),
            )
            if edge.id in edges:
                raise ExecutionError(f"Duplicate edge id: {edge.id}", code="23000")
            edges[edge.id] = edge
        return cls(name=str(data.get("name", "graph")), nodes=nodes, edges=edges)

    @classmethod
    def load(cls, path: str | Path) -> "PropertyGraph":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "nodes": [
                {
                    "id": node.id,
                    "labels": sorted(node.labels),
                    "properties": copy.deepcopy(node.properties),
                }
                for node in self.nodes.values()
            ],
            "edges": [
                {
                    "id": edge.id,
                    "type": edge.type,
                    "labels": sorted(edge.labels),
                    "directed": edge.directed,
                    "source": edge.source,
                    "target": edge.target,
                    "properties": copy.deepcopy(edge.properties),
                }
                for edge in self.edges.values()
            ],
        }

    def save(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    def clone(self) -> "PropertyGraph":
        return PropertyGraph(
            self.name,
            {node_id: node.clone() for node_id, node in self.nodes.items()},
            {edge_id: edge.clone() for edge_id, edge in self.edges.items()},
        )

    def replace_with(self, other: "PropertyGraph") -> None:
        if self.name != other.name:
            raise ExecutionError("Cannot replace a graph with a differently named graph")
        self.nodes = {node_id: node.clone() for node_id, node in other.nodes.items()}
        self.edges = {edge_id: edge.clone() for edge_id, edge in other.edges.items()}
        self.rebuild_indexes()

    def element(self, element_id: str | tuple[str, str]) -> GraphElement:
        if isinstance(element_id, tuple):
            kind, identifier = element_id
            if kind not in {"NODE", "EDGE"}:
                raise ExecutionError("Invalid element kind", code="22000")
            elements = self.nodes if kind == "NODE" else self.edges
            if identifier in elements:
                return elements[identifier]
            raise ExecutionError(f"Unknown {kind.lower()} identity")
        if element_id in self.nodes and element_id in self.edges:
            raise ExecutionError("Ambiguous element identity: supply its kind", code="22000")
        if element_id in self.nodes:
            return self.nodes[element_id]
        if element_id in self.edges:
            return self.edges[element_id]
        raise ExecutionError(f"Unknown graph element: {element_id}")

    def out_edges(self, node_id: str) -> Iterator[Edge]:
        for edge_id in self._out.get(node_id, []):
            yield self.edges[edge_id]

    def in_edges(self, node_id: str) -> Iterator[Edge]:
        for edge_id in self._in.get(node_id, []):
            yield self.edges[edge_id]

    def adjacent(self, node_id: str, direction: str = "out") -> Iterator[tuple[Edge, Node]]:
        if direction not in {"out", "in", "both", "undirected"}:
            raise ExecutionError("Unsupported edge direction", code="22000")
        seen = set()
        if direction in {"out", "both", "undirected"}:
            for edge in self.out_edges(node_id):
                if (edge.directed and direction != "undirected") or (
                    not edge.directed and direction in {"both", "undirected"}
                ):
                    seen.add(edge.id)
                    yield edge, self.nodes[edge.target]
        if direction in {"in", "both", "undirected"}:
            for edge in self.in_edges(node_id):
                if edge.id in seen:
                    continue  # A self-loop supplies one step, not two copies.
                if (edge.directed and direction != "undirected") or (
                    not edge.directed and direction in {"both", "undirected"}
                ):
                    yield edge, self.nodes[edge.source]

    def matching_nodes(
        self,
        labels: Iterable[str] = (),
        properties: dict[str, Any] | None = None,
    ) -> Iterator[Node]:
        required_labels = set(labels)
        required_properties = properties or {}
        for node in self.nodes.values():
            if not required_labels.issubset(node.labels):
                continue
            if all(node.properties.get(key) == value for key, value in required_properties.items()):
                yield node

    def add_node(self, node: Node) -> None:
        if node.id in self.nodes:
            raise ExecutionError(f"Duplicate graph element id: {node.id}", code="23000")
        self.nodes[node.id] = node
        self._out[node.id] = []
        self._in[node.id] = []

    def add_edge(self, edge: Edge) -> None:
        if edge.id in self.edges:
            raise ExecutionError(f"Duplicate graph element id: {edge.id}", code="23000")
        if edge.source not in self.nodes or edge.target not in self.nodes:
            raise ExecutionError("Inserted edge has a missing endpoint", code="G2000")
        self.edges[edge.id] = edge
        self._out.setdefault(edge.source, []).append(edge.id)
        self._in.setdefault(edge.target, []).append(edge.id)

    def delete_edge(self, edge_id: str) -> None:
        edge = self.edges.pop(edge_id)
        self._out[edge.source].remove(edge_id)
        self._in[edge.target].remove(edge_id)

    def delete_node(self, node_id: str, *, detach: bool = False) -> None:
        incident = list(self._out.get(node_id, [])) + list(self._in.get(node_id, []))
        if incident and not detach:
            raise ExecutionError(
                f"NODETACH DELETE cannot remove node {node_id!r} while incident edges remain",
                code="G1001",
            )
        for edge_id in dict.fromkeys(incident):
            if edge_id in self.edges:
                self.delete_edge(edge_id)
        self.nodes.pop(node_id)
        self._out.pop(node_id, None)
        self._in.pop(node_id, None)
