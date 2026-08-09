from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .catalog import AuthorizationCatalog
from .errors import AuthGQLError
from .model import PropertyGraph
from .session import AuthGQLSession, split_statements


def _parameter(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("parameters use NAME=JSON_VALUE")
    name, raw = text.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("parameter name is empty")
    try:
        return name, json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON parameter value: {exc}") from exc


def _emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=False))


def _run_statements(
    session: AuthGQLSession,
    statements: Iterable[str],
    *,
    keep_going: bool = False,
) -> int:
    status = 0
    for statement in statements:
        try:
            _emit(session.execute(statement))
        except AuthGQLError as exc:
            _emit(exc.to_dict())
            status = 1
            if not keep_going:
                break
        except Exception as exc:  # keep unexpected prototype faults machine-readable
            _emit({"status": "error", "code": "AUTHGQL-INTERNAL", "message": str(exc)})
            status = 1
            if not keep_going:
                break
    return status


def _repl(session: AuthGQLSession) -> int:
    print("AuthGQL research prototype; terminate statements with ';' and use QUIT; to exit.")
    buffer = ""
    while True:
        try:
            line = input("authgql> " if not buffer else "      -> ")
        except EOFError:
            print()
            return 0
        if not buffer and line.strip().upper() in {"QUIT", "QUIT;", "EXIT", "EXIT;"}:
            return 0
        buffer += ("\n" if buffer else "") + line
        if ";" not in line:
            continue
        statements = split_statements(buffer)
        buffer = ""
        _run_statements(session, statements, keep_going=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute the AuthGQL research subset over an in-memory property graph."
    )
    parser.add_argument("script", nargs="?", type=Path, help="semicolon-delimited script")
    parser.add_argument(
        "--graph",
        action="append",
        type=Path,
        default=[],
        help="property-graph JSON file; repeat to make multiple graphs available",
    )
    parser.add_argument("--catalog", type=Path, help="authorization-catalog JSON file")
    parser.add_argument("--user", default="admin", help="session authorization identifier")
    parser.add_argument("--working-graph", help="initial working graph")
    parser.add_argument(
        "--parameter",
        action="append",
        type=_parameter,
        default=[],
        metavar="NAME=JSON",
        help="dynamic request/session parameter",
    )
    parser.add_argument(
        "--command",
        action="append",
        default=[],
        help="execute one statement; may be repeated",
    )
    parser.add_argument(
        "--compare-reference",
        action="store_true",
        help="compare each read query with the authorized-view reference evaluator",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="continue a script after a failed statement",
    )
    parser.add_argument("--save-graph", type=Path, help="save the final working graph as JSON")
    parser.add_argument("--save-catalog", type=Path, help="save the final catalog as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    graphs: dict[str, PropertyGraph] = {}
    for path in args.graph:
        graph = PropertyGraph.load(path)
        if graph.name in graphs:
            raise SystemExit(f"duplicate graph name {graph.name!r} from {path}")
        graphs[graph.name] = graph
    if not graphs:
        graphs["default"] = PropertyGraph("default")
    catalog = (
        AuthorizationCatalog.load(args.catalog)
        if args.catalog
        else AuthGQLSession.bootstrap_catalog(args.user)
    )
    session = AuthGQLSession(
        graphs,
        catalog,
        user=args.user,
        graph_name=args.working_graph,
        parameters=dict(args.parameter),
        compare_reference=args.compare_reference,
    )

    statements: list[str] = []
    statements.extend(args.command)
    if args.script:
        statements.extend(split_statements(args.script.read_text(encoding="utf-8")))
    status = (
        _run_statements(session, statements, keep_going=args.keep_going)
        if statements
        else _repl(session)
    )

    if args.save_graph:
        session._active_graph().save(args.save_graph)
    if args.save_catalog:
        session.catalog.save(args.save_catalog)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
