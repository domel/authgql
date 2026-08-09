# Contributing

AuthGQL is a research prototype. Contributions should keep the implementation
and the documented supported fragment aligned.

Before opening a pull request, run from this directory:

```bash
python -m unittest discover -s tests -v
ruff check .
mypy authgql authgql_console.py evaluate.py
python evaluate.py
```

Please include tests for changes to authorization semantics, preserve the
fail-closed behavior, and update `README.md` when the supported syntax,
limitations, or evaluation procedure changes. The project intentionally uses
only the Python standard library at runtime.
