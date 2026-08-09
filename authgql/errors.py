from __future__ import annotations


class AuthGQLError(Exception):
    """Base exception carrying a GQL-style status code."""

    def __init__(self, message: str, code: str = "42000") -> None:
        super().__init__(message)
        self.code = code

    def to_dict(self) -> dict[str, str]:
        return {"status": "error", "code": self.code, "message": str(self)}


class ParseError(AuthGQLError):
    pass


class AuthorizationError(AuthGQLError):
    pass


class CatalogError(AuthGQLError):
    pass


class ExecutionError(AuthGQLError):
    pass
