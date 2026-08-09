from __future__ import annotations

from typing import Any

from .catalog import AuthorizationCatalog
from .errors import AuthorizationError
from .metrics import ExecutionMetrics
from .model import GraphElement, PropertyGraph
from .policy import PolicyEvaluationContext, compile_predicate


class AuthorizationEngine:
    def __init__(
        self,
        catalog: AuthorizationCatalog,
        graph: PropertyGraph,
        user: str,
        parameters: dict[str, Any] | None = None,
        metrics: ExecutionMetrics | None = None,
    ) -> None:
        self.catalog = catalog
        self.graph = graph
        self.user = user
        self.parameters = parameters or {}
        self.metrics = metrics or ExecutionMetrics()
        self._compiled_using = {
            policy.name: compile_predicate(policy.using) for policy in catalog.policies
        }
        self._compiled_with = {
            policy.name: compile_predicate(policy.with_check) for policy in catalog.policies
        }

    @property
    def roles(self) -> set[str]:
        return self.catalog.effective_roles(self.user)

    def _object_check(
        self,
        action: str,
        resource: GraphElement | None = None,
        property_name: str | None = None,
    ) -> None:
        self.metrics.object_checks += 1
        if property_name is not None:
            self.metrics.property_checks += 1
        permitted = self.catalog.object_permitted(
            self.user, action, self.graph.name, resource, property_name
        )
        if not permitted:
            self.metrics.object_denials += 1
            target = self.graph.name
            if resource is not None:
                target += f"/{resource.kind.lower()}"
            if property_name is not None:
                target += f".{property_name}"
            raise AuthorizationError(
                f"Authorization identifier {self.user!r} lacks {action.upper()} on {target}",
                code="42000",
            )

    def check_access(self) -> None:
        self._object_check("ACCESS")

    def _fine_decision(
        self,
        action: str,
        selector_resource: GraphElement,
        graph: PropertyGraph,
        context_resource: GraphElement | None,
        new_resource: GraphElement | None = None,
        phase: str = "using",
    ) -> bool:
        policies = self.catalog.applicable_policies(
            self.user, action, self.graph.name, selector_resource
        )
        if not policies:
            return True
        context = PolicyEvaluationContext(
            user=self.user,
            roles=self.roles,
            parameters=self.parameters,
            resource=context_resource,
            new_resource=new_resource,
        )
        evaluated: list[tuple[str, bool | None]] = []
        for policy in policies:
            self.metrics.policy_evaluations += 1
            if phase == "with_check":
                self.metrics.with_check_evaluations += 1
                predicate = self._compiled_with[policy.name]
            else:
                predicate = self._compiled_using[policy.name]
            try:
                result = predicate(graph, context)
                if result is not True and result is not False and result is not None:
                    raise TypeError(
                        "policy predicate must evaluate to TRUE, FALSE, or UNKNOWN"
                    )
            except Exception:  # fail closed without exposing predicate inputs
                raise AuthorizationError(
                    "Authorization policy evaluation failed", code="42000"
                ) from None
            evaluated.append((policy.effect, result))
        if any(
            effect == "DENY" and result is not False
            for effect, result in evaluated
        ):
            self.metrics.policy_denials += 1
            return False
        if any(
            effect == "PERMIT" and result is True
            for effect, result in evaluated
        ):
            return True
        self.metrics.policy_denials += 1
        return False

    def check(
        self,
        action: str,
        resource: GraphElement,
        property_name: str | None = None,
        graph: PropertyGraph | None = None,
    ) -> None:
        self._object_check(action, resource, property_name)
        evaluation_graph = graph or self.graph
        if not self._fine_decision(
            action,
            resource,
            evaluation_graph,
            context_resource=resource,
            phase="using",
        ):
            raise AuthorizationError(
                f"Fine-grained policy denied {action.upper()} on a {resource.kind.lower()}",
                code="42000",
            )

    def permits(
        self,
        action: str,
        resource: GraphElement,
        property_name: str | None = None,
        graph: PropertyGraph | None = None,
    ) -> bool:
        try:
            self.check(action, resource, property_name, graph)
            return True
        except AuthorizationError:
            return False

    def check_with(
        self,
        action: str,
        old_resource: GraphElement | None,
        new_resource: GraphElement,
        prospective_graph: PropertyGraph,
        property_name: str | None = None,
    ) -> None:
        self._object_check(action, new_resource, property_name)
        if not self._fine_decision(
            action,
            new_resource=new_resource,
            selector_resource=new_resource,
            graph=prospective_graph,
            context_resource=old_resource,
            phase="with_check",
        ):
            raise AuthorizationError(
                f"WITH CHECK denied {action.upper()} on a {new_resource.kind.lower()}",
                code="42000",
            )

    def read_property(
        self,
        resource: GraphElement,
        property_name: str,
        graph: PropertyGraph | None = None,
    ) -> Any:
        self.check("READ", resource, property_name, graph=graph)
        return resource.properties.get(property_name)
