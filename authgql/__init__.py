"""AuthGQL research prototype."""

from .catalog import AuthorizationCatalog, PolicyDescriptor, PrivilegeFact, Target
from .executor import ExecutionResult, SecureExecutor
from .model import Edge, Node, PropertyGraph
from .parser import parse_query
from .session import AuthGQLSession

__all__ = [
    "AuthGQLSession",
    "AuthorizationCatalog",
    "Edge",
    "ExecutionResult",
    "Node",
    "PolicyDescriptor",
    "PrivilegeFact",
    "PropertyGraph",
    "SecureExecutor",
    "Target",
    "parse_query",
]
