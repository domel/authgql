# AuthGQL

AuthGQL is an in-memory reference implementation of graph authorization with
GQL-style patterns, role administration, `TRAVERSE` and `READ` privileges, and
old-state `USING` and proposed-state `WITH CHECK` policies.

Publication record: [https://doi.org/10.6084/m9.figshare.33544702](https://doi.org/10.6084/m9.figshare.33544702)

## Requirements

- Python 3.10 or later
- No third-party runtime packages

Clone the repository and run commands from its root directory. An isolated
environment is optional:

```sh
python -m venv .venv
. .venv/bin/activate
```

## Use the command line

Run a query against the included hospital graph:

```sh
python -m authgql \
  --graph examples/hospital.graph.json \
  --catalog examples/hospital.catalog.json \
  --user alice \
  --command 'MATCH (p:Patient) RETURN p.age AS age'
```

Without `--command`, the program starts a semicolon-terminated console. Use
`--parameter NAME=JSON` for parameters, `--working-graph NAME` to select a
graph, and `--save-graph PATH` or `--save-catalog PATH` to export JSON.
`START TRANSACTION`, `COMMIT`, and `ROLLBACK` control staged changes.

## Python interface

The package is available under `authgql`. The main modules are:

- `model.py`: graph elements and graph operations
- `catalog.py`: roles, grants, denials, policies, and snapshots
- `parser.py` and `validation.py`: supported query syntax and checks
- `analyzer.py`: static authorization obligations
- `authorization.py` and `policy.py`: privilege and policy decisions
- `executor.py` and `session.py`: guarded execution and transactions

## Verification

Run the test suite and the supplied hospital workload:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 python evaluate.py
```

Run the static checks when the development tools are installed:

```sh
ruff check .
mypy authgql authgql_console.py evaluate.py
```

The implementation supports a documented GQL fragment. It is not a complete
GQL database server and does not provide authentication, durable recovery,
concurrent execution, or storage-level access isolation.

## License

MIT. See [LICENSE](LICENSE).
