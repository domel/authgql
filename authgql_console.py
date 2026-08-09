#!/usr/bin/env python3
"""Compatibility entry point for the modular AuthGQL prototype."""

from authgql.cli import main
from authgql.errors import AuthorizationError as AuthError, ParseError
from authgql.session import AuthGQLSession as Engine

__all__ = ["AuthError", "Engine", "ParseError", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
