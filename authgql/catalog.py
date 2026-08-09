from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
import copy
import json

from .errors import CatalogError
from .model import Edge, GraphElement, Node


UPDATE_ACTIONS = {"INSERT", "DELETE", "SET", "REMOVE"}
POLICY_ACTIONS = {"TRAVERSE", "READ", "DELETE", "INSERT", "SET", "REMOVE"}
ACTION_IMPLICATIONS: dict[str, set[str]] = {
    "MATCH": {"MATCH", "TRAVERSE", "READ"},
    "UPDATE": {"UPDATE", *UPDATE_ACTIONS},
    "ADMINISTER": {"ADMINISTER", "CREATE", "ALTER", "DROP"},
}


def action_implies(granted: str, required: str) -> bool:
    granted = granted.upper()
    required = required.upper()
    return required == granted or required in ACTION_IMPLICATIONS.get(granted, set())


def policy_action_matches(policy_actions: Iterable[str], required: str) -> bool:
    required = required.upper()
    actions = {action.upper() for action in policy_actions}
    return required in actions


@dataclass(frozen=True)
class Target:
    kind: str
    graph: str
    labels: frozenset[str] = frozenset()
    edge_types: frozenset[str] = frozenset()
    properties: frozenset[str] = frozenset()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Target":
        return cls(
            kind=str(raw.get("kind", "GRAPH")).upper(),
            graph=str(raw["graph"]),
            labels=frozenset(map(str, raw.get("labels", []))),
            edge_types=frozenset(map(str, raw.get("edge_types", raw.get("types", [])))),
            properties=frozenset(map(str, raw.get("properties", []))),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": self.kind, "graph": self.graph}
        if self.labels:
            result["labels"] = sorted(self.labels)
        if self.edge_types:
            result["edge_types"] = sorted(self.edge_types)
        if self.properties:
            result["properties"] = sorted(self.properties)
        return result

    def applies_to_resource(
        self,
        graph_name: str,
        resource: GraphElement | None,
        property_name: str | None = None,
    ) -> bool:
        if self.graph not in {graph_name, "*"}:
            return False
        if self.kind == "GRAPH":
            return True
        if resource is None:
            return False
        if self.kind == "NODES":
            return isinstance(resource, Node) and (
                not self.labels or bool(self.labels.intersection(resource.labels))
            )
        if self.kind == "EDGES":
            return isinstance(resource, Edge) and (
                not self.edge_types or resource.type in self.edge_types
            )
        if self.kind == "PROPERTIES":
            if property_name is None:
                return False
            if self.properties and property_name not in self.properties:
                return False
            if isinstance(resource, Node):
                if self.edge_types:
                    return False
                return not self.labels or bool(self.labels.intersection(resource.labels))
            if isinstance(resource, Edge):
                if self.labels:
                    return False
                return not self.edge_types or resource.type in self.edge_types
            return False
        return False

    def covers_target(self, required: "Target") -> bool:
        if self.graph != "*" and self.graph != required.graph:
            return False
        if self.kind == "GRAPH":
            return True

        def covers_classes(granted: frozenset[str], needed: frozenset[str]) -> bool:
            # Label/type target sets denote a union of element classes.  A
            # pattern requiring A:B is contained in a grant for A because every
            # matching element belongs to the A class.  An empty granted class
            # denotes all classes, whereas an empty required class is unbounded.
            return not granted or bool(granted.intersection(needed))

        def covers_properties(
            granted: frozenset[str], needed: frozenset[str]
        ) -> bool:
            # Property keys are not alternatives: every statically required key
            # must be present in the grant (an empty grant denotes all keys).
            return not granted or needed.issubset(granted)

        if self.kind == "NODES":
            if required.kind not in {"NODES", "PROPERTIES"} or required.edge_types:
                return False
            return covers_classes(self.labels, required.labels)
        if self.kind == "EDGES":
            if required.kind not in {"EDGES", "PROPERTIES"} or required.labels:
                return False
            return covers_classes(self.edge_types, required.edge_types)
        if self.kind == "PROPERTIES":
            if required.kind != "PROPERTIES":
                return False
            labels_ok = covers_classes(self.labels, required.labels)
            types_ok = covers_classes(self.edge_types, required.edge_types)
            properties_ok = covers_properties(self.properties, required.properties)
            return labels_ok and types_ok and properties_ok
        return False


@dataclass(frozen=True)
class PrivilegeFact:
    grantee: str
    effect: str
    actions: frozenset[str]
    target: Target
    grant_option: bool = False
    grantor: str | None = None
    grantee_kind: str = "AUTO"

    @classmethod
    def from_dict(
        cls,
        raw: dict[str, Any],
        role_names: Iterable[str] = (),
    ) -> "PrivilegeFact":
        actions = raw.get("actions")
        if actions is None:
            actions = [raw["action"]]
        grantee = str(raw["grantee"])
        explicit_kind = raw.get("grantee_kind")
        grantee_kind = (
            str(explicit_kind).upper()
            if explicit_kind is not None
            else ("ROLE" if grantee in set(role_names) else "USER")
        )
        return cls(
            grantee=grantee,
            effect=str(raw.get("effect", "PERMIT")).upper(),
            actions=frozenset(str(action).upper() for action in actions),
            target=Target.from_dict(raw["target"]),
            grant_option=bool(raw.get("grant_option", False)),
            grantor=raw.get("grantor"),
            grantee_kind=grantee_kind,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "grantee": self.grantee,
            "grantee_kind": self.grantee_kind,
            "effect": self.effect,
            "actions": sorted(self.actions),
            "target": self.target.to_dict(),
            "grant_option": self.grant_option,
            **({"grantor": self.grantor} if self.grantor else {}),
        }


@dataclass
class PolicyDescriptor:
    name: str
    graph: str
    actions: set[str]
    grantees: set[str]
    effect: str
    selector: dict[str, Any] = field(default_factory=dict)
    using: Any = True
    with_check: Any = True
    dependencies: set[str] = field(default_factory=set)
    owner: str = "system"
    enabled: bool = True
    grantee_kinds: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        raw: dict[str, Any],
        role_names: Iterable[str] = (),
    ) -> "PolicyDescriptor":
        grantees = set(map(str, raw.get("grantees", [])))
        raw_kinds = {
            str(name): str(kind).upper()
            for name, kind in raw.get("grantee_kinds", {}).items()
        }
        known_roles = set(role_names)
        return cls(
            name=str(raw["name"]),
            graph=str(raw["graph"]),
            actions={str(action).upper() for action in raw.get("actions", [])},
            grantees=grantees,
            effect=str(raw.get("effect", "PERMIT")).upper(),
            selector=copy.deepcopy(raw.get("selector", {})),
            using=copy.deepcopy(raw.get("using", True)),
            with_check=copy.deepcopy(raw.get("with_check", True)),
            dependencies=set(map(str, raw.get("dependencies", []))),
            owner=str(raw.get("owner", "system")),
            enabled=bool(raw.get("enabled", True)),
            grantee_kinds={
                grantee: raw_kinds.get(
                    grantee, "ROLE" if grantee in known_roles else "USER"
                )
                for grantee in grantees
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "graph": self.graph,
            "actions": sorted(self.actions),
            "grantees": sorted(self.grantees),
            "grantee_kinds": {
                grantee: self.grantee_kinds.get(grantee, "AUTO")
                for grantee in sorted(self.grantees)
            },
            "effect": self.effect,
            "selector": copy.deepcopy(self.selector),
            "using": copy.deepcopy(self.using),
            "with_check": copy.deepcopy(self.with_check),
            "dependencies": sorted(self.dependencies),
            "owner": self.owner,
            "enabled": self.enabled,
        }

    def selector_matches(self, resource: GraphElement) -> bool:
        kind = str(self.selector.get("kind", "ANY")).upper()
        if kind == "NODE" and not isinstance(resource, Node):
            return False
        if kind == "EDGE" and not isinstance(resource, Edge):
            return False
        labels = set(map(str, self.selector.get("labels", [])))
        if labels and (not isinstance(resource, Node) or not labels.intersection(resource.labels)):
            return False
        edge_types = set(map(str, self.selector.get("edge_types", self.selector.get("types", []))))
        if edge_types and (not isinstance(resource, Edge) or resource.type not in edge_types):
            return False
        return True


@dataclass
class AuthorizationCatalog:
    roles: dict[str, set[str]] = field(default_factory=dict)
    user_roles: dict[str, set[str]] = field(default_factory=dict)
    privileges: list[PrivilegeFact] = field(default_factory=list)
    policies: list[PolicyDescriptor] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    revision: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuthorizationCatalog":
        raw_roles = data.get("roles", {})
        roles: dict[str, set[str]] = {}
        if isinstance(raw_roles, list):
            roles = {str(name): set() for name in raw_roles}
        else:
            for name, definition in raw_roles.items():
                if isinstance(definition, dict):
                    roles[str(name)] = set(map(str, definition.get("inherits", [])))
                else:
                    roles[str(name)] = set(map(str, definition or []))

        catalog = cls(
            roles=roles,
            user_roles={
                str(user): set(map(str, assigned))
                for user, assigned in data.get("user_roles", {}).items()
            },
            privileges=[
                PrivilegeFact.from_dict(item, roles)
                for item in data.get("privileges", [])
            ],
            policies=[
                PolicyDescriptor.from_dict(item, roles)
                for item in data.get("policies", [])
            ],
            metadata=copy.deepcopy(data.get("metadata", {})),
            revision=int(data.get("revision", 0)),
        )
        catalog.validate()
        return catalog

    @classmethod
    def load(cls, path: str | Path) -> "AuthorizationCatalog":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def to_dict(self) -> dict[str, Any]:
        privilege_rows: list[dict[str, Any]] = []
        for fact in self.privileges:
            row = fact.to_dict()
            if row["grantee_kind"] == "AUTO":
                row["grantee_kind"] = (
                    "ROLE" if fact.grantee in self.roles else "USER"
                )
            privilege_rows.append(row)

        policy_rows: list[dict[str, Any]] = []
        for policy in self.policies:
            row = policy.to_dict()
            row["grantee_kinds"] = {
                grantee: policy.grantee_kinds.get(
                    grantee, "ROLE" if grantee in self.roles else "USER"
                )
                for grantee in sorted(policy.grantees)
            }
            policy_rows.append(row)

        return {
            "roles": {
                role: {"inherits": sorted(parents)} for role, parents in sorted(self.roles.items())
            },
            "user_roles": {
                user: sorted(roles) for user, roles in sorted(self.user_roles.items())
            },
            "privileges": privilege_rows,
            "policies": policy_rows,
            "metadata": copy.deepcopy(self.metadata),
            "revision": self.revision,
        }

    def clone(self) -> "AuthorizationCatalog":
        """Return an isolated authorization-state snapshot."""

        return AuthorizationCatalog.from_dict(self.to_dict())

    def replace_with(self, other: "AuthorizationCatalog") -> None:
        """Atomically replace this catalog's visible contents."""

        replacement = other.clone()
        replacement.validate()
        self.roles = replacement.roles
        self.user_roles = replacement.user_roles
        self.privileges = replacement.privileges
        self.policies = replacement.policies
        self.metadata = replacement.metadata
        self.revision = replacement.revision

    def touch(self) -> None:
        self.revision += 1

    def save(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    def validate(self) -> None:
        for role, inherited in self.roles.items():
            missing = inherited.difference(self.roles)
            if missing:
                raise CatalogError(f"Role {role!r} inherits unknown roles: {sorted(missing)}")
        self._validate_acyclic_roles()
        names = [policy.name for policy in self.policies]
        if len(names) != len(set(names)):
            raise CatalogError("Policy names must be unique")
        self._validate_policy_dependencies()
        for index, fact in enumerate(self.privileges):
            if fact.grantee_kind == "AUTO":
                fact = PrivilegeFact(
                    grantee=fact.grantee,
                    effect=fact.effect,
                    actions=fact.actions,
                    target=fact.target,
                    grant_option=fact.grant_option,
                    grantor=fact.grantor,
                    grantee_kind="ROLE" if fact.grantee in self.roles else "USER",
                )
                self.privileges[index] = fact
            if fact.grantee_kind not in {"USER", "ROLE"}:
                raise CatalogError(f"Invalid privilege grantee kind: {fact.grantee_kind!r}")
            if fact.grantee_kind == "ROLE" and fact.grantee not in self.roles:
                raise CatalogError(f"Privilege names unknown role: {fact.grantee!r}")
            if fact.grant_option and fact.grantee_kind != "USER":
                raise CatalogError("WITH GRANT OPTION is supported only for a USER grantee")
            if fact.effect == "DENY" and fact.grant_option:
                raise CatalogError("A denial cannot carry WITH GRANT OPTION")
            if fact.effect not in {"PERMIT", "DENY"}:
                raise CatalogError(f"Invalid privilege effect: {fact.effect!r}")
            if not fact.actions:
                raise CatalogError("A privilege fact must contain at least one action")
        for policy in self.policies:
            if policy.effect not in {"PERMIT", "DENY"}:
                raise CatalogError(f"Invalid policy effect on {policy.name!r}: {policy.effect!r}")
            if not policy.actions:
                raise CatalogError(f"Policy {policy.name!r} must contain at least one action")
            unsupported_actions = policy.actions.difference(POLICY_ACTIONS)
            if unsupported_actions:
                raise CatalogError(
                    f"Unsupported policy actions on {policy.name!r}: "
                    f"{sorted(unsupported_actions)}"
                )
            if not policy.grantees:
                raise CatalogError(f"Policy {policy.name!r} must contain at least one grantee")
            for grantee in policy.grantees:
                kind = policy.grantee_kinds.get(
                    grantee, "ROLE" if grantee in self.roles else "USER"
                ).upper()
                policy.grantee_kinds[grantee] = kind
                if kind not in {"USER", "ROLE"}:
                    raise CatalogError(
                        f"Invalid policy grantee kind on {policy.name!r}: {kind!r}"
                    )
                if kind == "ROLE" and grantee not in self.roles:
                    raise CatalogError(
                        f"Policy {policy.name!r} names unknown role {grantee!r}"
                    )
            policy.grantee_kinds = {
                grantee: policy.grantee_kinds[grantee]
                for grantee in policy.grantees
            }
        for user, roles in self.user_roles.items():
            missing = roles.difference(self.roles)
            if missing:
                raise CatalogError(f"User {user!r} has unknown roles: {sorted(missing)}")

    def _validate_acyclic_roles(self) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(role: str) -> None:
            if role in visiting:
                raise CatalogError(f"Role inheritance cycle detected at {role!r}")
            if role in visited:
                return
            visiting.add(role)
            for parent in self.roles.get(role, set()):
                visit(parent)
            visiting.remove(role)
            visited.add(role)

        for role in self.roles:
            visit(role)

    def _validate_policy_dependencies(self) -> None:
        policies = {policy.name: policy for policy in self.policies}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visiting:
                raise CatalogError(f"Policy dependency cycle detected at {name!r}")
            if name in visited or name not in policies:
                return
            visiting.add(name)
            for dependency in policies[name].dependencies:
                if dependency.startswith("policy:"):
                    dependency_name = dependency.split(":", 1)[1]
                    if dependency_name not in policies:
                        raise CatalogError(
                            f"Policy {name!r} depends on unknown policy {dependency_name!r}"
                        )
                    visit(dependency_name)
            visiting.remove(name)
            visited.add(name)

        for name in policies:
            visit(name)

    def effective_roles(self, user: str) -> set[str]:
        result: set[str] = set()
        stack = list(self.user_roles.get(user, set()))
        while stack:
            role = stack.pop()
            if role in result:
                continue
            result.add(role)
            stack.extend(self.roles.get(role, set()))
        return result

    def principals_for(self, user: str) -> set[str]:
        return {user, *self.effective_roles(user)}

    def principal_keys_for(self, user: str) -> set[tuple[str, str]]:
        return {("USER", user), *(("ROLE", role) for role in self.effective_roles(user))}

    def applicable_privileges(
        self,
        user: str,
        action: str,
        graph_name: str,
        resource: GraphElement | None = None,
        property_name: str | None = None,
    ) -> list[PrivilegeFact]:
        principals = self.principal_keys_for(user)
        result: list[PrivilegeFact] = []
        for fact in self.privileges:
            if (fact.grantee_kind, fact.grantee) not in principals:
                continue
            if not any(action_implies(granted, action) for granted in fact.actions):
                continue
            if fact.target.applies_to_resource(graph_name, resource, property_name):
                result.append(fact)
        return result

    def object_permitted(
        self,
        user: str,
        action: str,
        graph_name: str,
        resource: GraphElement | None = None,
        property_name: str | None = None,
    ) -> bool:
        facts = self.applicable_privileges(user, action, graph_name, resource, property_name)
        if any(fact.effect == "DENY" for fact in facts):
            return False
        return any(fact.effect == "PERMIT" for fact in facts)

    def object_target_permitted(self, user: str, action: str, target: Target) -> bool:
        principals = self.principal_keys_for(user)
        facts = [
            fact
            for fact in self.privileges
            if (fact.grantee_kind, fact.grantee) in principals
            and any(action_implies(granted, action) for granted in fact.actions)
            and fact.target.covers_target(target)
        ]
        if any(fact.effect == "DENY" for fact in facts):
            return False
        return any(fact.effect == "PERMIT" for fact in facts)

    def delegation_permitted(self, user: str, action: str, target: Target) -> bool:
        """Return whether *user* may delegate an action on a target.

        ADMINISTER is an explicit administrative bypass. Otherwise a matching
        permit with grant option is required and any matching denial wins.
        """

        effective_principals = self.principal_keys_for(user)
        effective_facts = [
            fact
            for fact in self.privileges
            if (fact.grantee_kind, fact.grantee) in effective_principals
            and any(action_implies(granted, action) for granted in fact.actions)
            and fact.target.covers_target(target)
        ]
        if self.object_target_permitted(user, "ADMINISTER", target):
            return True
        if any(fact.effect == "DENY" for fact in effective_facts):
            return False
        return any(
            fact.effect == "PERMIT"
            and fact.grant_option
            and fact.grantee_kind == "USER"
            and fact.grantee == user
            for fact in effective_facts
        )

    def applicable_policies(
        self,
        user: str,
        action: str,
        graph_name: str,
        resource: GraphElement,
    ) -> list[PolicyDescriptor]:
        principals = self.principal_keys_for(user)
        return [
            policy
            for policy in self.policies
            if policy.enabled
            and policy.graph == graph_name
            and any(
                (policy.grantee_kinds[grantee], grantee) in principals
                for grantee in policy.grantees
            )
            and policy_action_matches(policy.actions, action)
            and policy.selector_matches(resource)
        ]

    # ---------- Transaction-friendly administration ----------

    def create_role(self, name: str) -> None:
        if name in self.roles:
            raise CatalogError(f"Role already exists: {name!r}")
        self.roles[name] = set()
        self.touch()

    def grant_role(self, role: str, member: str, member_kind: str = "USER") -> None:
        if role not in self.roles:
            raise CatalogError(f"Unknown role: {role!r}")
        member_kind = member_kind.upper()
        if member_kind == "USER":
            self.user_roles.setdefault(member, set()).add(role)
        elif member_kind == "ROLE":
            if member not in self.roles:
                raise CatalogError(f"Unknown member role: {member!r}")
            self.roles[member].add(role)
        else:
            raise CatalogError(f"Unsupported role member kind: {member_kind!r}")
        try:
            self.validate()
        except Exception:
            if member_kind == "USER":
                self.user_roles.get(member, set()).discard(role)
            else:
                self.roles[member].discard(role)
            raise
        self.touch()

    def revoke_role(self, role: str, member: str, member_kind: str = "USER") -> None:
        member_kind = member_kind.upper()
        assignments = self.user_roles if member_kind == "USER" else self.roles
        if member not in assignments or role not in assignments[member]:
            raise CatalogError(f"Role {role!r} is not granted to {member_kind.lower()} {member!r}")
        assignments[member].remove(role)
        self.touch()

    def add_privilege(self, fact: PrivilegeFact) -> None:
        if fact.grantee_kind == "AUTO":
            fact = PrivilegeFact(
                grantee=fact.grantee,
                effect=fact.effect,
                actions=fact.actions,
                target=fact.target,
                grant_option=fact.grant_option,
                grantor=fact.grantor,
                grantee_kind="ROLE" if fact.grantee in self.roles else "USER",
            )
        if fact in self.privileges:
            return
        self.privileges.append(fact)
        self.validate()
        self.touch()

    def revoke_privilege(
        self,
        grantee: str,
        actions: Iterable[str],
        target: Target,
        *,
        grantee_kind: str = "USER",
        effect: str = "PERMIT",
        grant_option_only: bool = False,
        behavior: str = "RESTRICT",
    ) -> int:
        """Revoke exact-target privilege facts and honor dependent grants.

        Dependency tracking uses the ``grantor`` recorded on delegated facts and
        the supporting action/target.  It is intentionally conservative when a
        dependent fact might also have another matching support: CASCADE still
        removes the recorded grant chain.
        """

        wanted = {action.upper() for action in actions}
        behavior = behavior.upper()
        grantee_kind = grantee_kind.upper()
        effect = effect.upper()
        if behavior not in {"RESTRICT", "CASCADE"}:
            raise CatalogError(f"Unsupported revoke behavior: {behavior!r}")
        if effect not in {"PERMIT", "DENY"}:
            raise CatalogError(f"Unsupported revoke effect: {effect!r}")
        if grant_option_only and effect != "PERMIT":
            raise CatalogError("A denial has no grant option")

        selected = [
            fact
            for fact in self.privileges
            if fact.grantee == grantee
            and fact.grantee_kind == grantee_kind
            and fact.target == target
            and bool(fact.actions.intersection(wanted))
            and fact.effect == effect
            and (not grant_option_only or fact.grant_option)
        ]
        if not selected:
            noun = "denial" if effect == "DENY" else "grant"
            raise CatalogError(f"No matching privilege {noun} exists")

        def supported_actions(
            fact: PrivilegeFact,
            support_grantor: str,
            support_actions: Iterable[str],
            support_target: Target,
        ) -> frozenset[str]:
            if (
                fact.effect != "PERMIT"
                or fact.grantor != support_grantor
                or not support_target.covers_target(fact.target)
            ):
                return frozenset()
            return frozenset(
                delegated_action
                for delegated_action in fact.actions
                if any(
                    action_implies(support_action, delegated_action)
                    for support_action in support_actions
                )
            )

        initial_supports: list[tuple[str, frozenset[str], Target]] = []
        if effect == "PERMIT" and grantee_kind == "USER":
            for fact in selected:
                if fact.grant_option:
                    initial_supports.append(
                        (grantee, frozenset(fact.actions.intersection(wanted)), fact.target)
                    )

        selected_ids = {id(fact) for fact in selected}
        dependent_actions: dict[int, set[str]] = {}
        frontier = list(initial_supports)
        while frontier:
            support_grantor, support_actions, support_target = frontier.pop()
            for candidate in self.privileges:
                candidate_id = id(candidate)
                if candidate_id in selected_ids:
                    continue
                newly_dependent = supported_actions(
                    candidate, support_grantor, support_actions, support_target
                ).difference(dependent_actions.get(candidate_id, set()))
                if not newly_dependent:
                    continue
                dependent_actions.setdefault(candidate_id, set()).update(newly_dependent)
                if candidate.grant_option and candidate.grantee_kind == "USER":
                    frontier.append(
                        (
                            candidate.grantee,
                            frozenset(newly_dependent),
                            candidate.target,
                        )
                    )

        if dependent_actions and behavior == "RESTRICT":
            raise CatalogError(
                f"RESTRICT rejected revocation because {grantee!r} has dependent grants"
            )

        def copy_fact(
            fact: PrivilegeFact,
            fact_actions: Iterable[str],
            *,
            grant_option: bool | None = None,
        ) -> PrivilegeFact:
            return PrivilegeFact(
                grantee=fact.grantee,
                effect=fact.effect,
                actions=frozenset(fact_actions),
                target=fact.target,
                grant_option=(
                    fact.grant_option if grant_option is None else grant_option
                ),
                grantor=fact.grantor,
                grantee_kind=fact.grantee_kind,
            )

        rebuilt: list[PrivilegeFact] = []
        changed = 0
        for fact in self.privileges:
            fact_id = id(fact)
            if fact_id not in selected_ids:
                removed_actions = (
                    dependent_actions.get(fact_id, set())
                    if behavior == "CASCADE"
                    else set()
                )
                if not removed_actions:
                    rebuilt.append(fact)
                    continue
                remaining = fact.actions.difference(removed_actions)
                if remaining:
                    rebuilt.append(copy_fact(fact, remaining))
                changed += 1
                continue

            changed += 1
            affected = fact.actions.intersection(wanted)
            remaining = fact.actions.difference(affected)
            if grant_option_only:
                rebuilt.append(copy_fact(fact, affected, grant_option=False))
                if remaining:
                    rebuilt.append(copy_fact(fact, remaining))
                continue
            if remaining:
                rebuilt.append(copy_fact(fact, remaining))

        self.privileges = rebuilt
        self.touch()
        return changed

    def add_policy(self, policy: PolicyDescriptor) -> None:
        if any(existing.name == policy.name for existing in self.policies):
            raise CatalogError(f"Policy already exists: {policy.name!r}")
        self.policies.append(policy)
        try:
            self.validate()
        except Exception:
            self.policies.pop()
            raise
        self.touch()

    def set_policy_enabled(self, name: str, enabled: bool) -> None:
        policy = next((item for item in self.policies if item.name == name), None)
        if policy is None:
            raise CatalogError(f"Unknown policy: {name!r}")
        policy.enabled = enabled
        self.touch()

    def drop_policy(self, name: str, behavior: str = "RESTRICT") -> list[str]:
        behavior = behavior.upper()
        if behavior not in {"RESTRICT", "CASCADE"}:
            raise CatalogError(f"Unsupported drop behavior: {behavior!r}")
        policies = {policy.name: policy for policy in self.policies}
        if name not in policies:
            raise CatalogError(f"Unknown policy: {name!r}")
        dependents = {
            policy.name
            for policy in self.policies
            if f"policy:{name}" in policy.dependencies
        }
        if dependents and behavior == "RESTRICT":
            raise CatalogError(
                f"RESTRICT rejected drop; dependent policies: {sorted(dependents)}"
            )
        removed = {name}
        if behavior == "CASCADE":
            while True:
                newly_dependent = {
                    policy.name
                    for policy in self.policies
                    if any(f"policy:{item}" in policy.dependencies for item in removed)
                }
                if newly_dependent.issubset(removed):
                    break
                removed.update(newly_dependent)
        self.policies = [policy for policy in self.policies if policy.name not in removed]
        self.touch()
        return sorted(removed)

    def drop_role(self, name: str, behavior: str = "RESTRICT") -> None:
        behavior = behavior.upper()
        if name not in self.roles:
            raise CatalogError(f"Unknown role: {name!r}")
        dependent_roles = {role for role, parents in self.roles.items() if name in parents}
        dependent_users = {user for user, roles in self.user_roles.items() if name in roles}
        dependent_privileges = [
            fact
            for fact in self.privileges
            if fact.grantee_kind == "ROLE" and fact.grantee == name
        ]
        dependent_policies = [
            policy
            for policy in self.policies
            if name in policy.grantees and policy.grantee_kinds.get(name) == "ROLE"
        ]
        if behavior == "RESTRICT" and (
            dependent_roles or dependent_users or dependent_privileges or dependent_policies
        ):
            raise CatalogError(f"RESTRICT rejected drop of role {name!r}; dependencies exist")
        if behavior not in {"RESTRICT", "CASCADE"}:
            raise CatalogError(f"Unsupported drop behavior: {behavior!r}")
        for parents in self.roles.values():
            parents.discard(name)
        for roles in self.user_roles.values():
            roles.discard(name)
        self.privileges = [
            fact
            for fact in self.privileges
            if not (fact.grantee_kind == "ROLE" and fact.grantee == name)
        ]
        for policy in dependent_policies:
            policy.grantees.discard(name)
            policy.grantee_kinds.pop(name, None)
        self.policies = [policy for policy in self.policies if policy.grantees]
        del self.roles[name]
        self.touch()

    def binding_tables(self, user: str, full: bool = False) -> dict[str, Any]:
        """Expose portable, read-only authorization catalog views."""

        effective = self.effective_roles(user)
        privilege_principals = self.principal_keys_for(user)
        policy_principals = self.principal_keys_for(user)
        privileges = self.privileges if full else [
            fact
            for fact in self.privileges
            if (fact.grantee_kind, fact.grantee) in privilege_principals
        ]
        policies = self.policies if full else [
            policy
            for policy in self.policies
            if any(
                (policy.grantee_kinds[grantee], grantee) in policy_principals
                for grantee in policy.grantees
            )
        ]
        return {
            "AUTHORIZATION_ROLES": [
                {"role": role, "inherits": sorted(parents)}
                for role, parents in sorted(self.roles.items())
                if full or role in effective
            ],
            "AUTHORIZATION_ROLE_MEMBERS": [
                {"authorization_identifier": member, "role": role}
                for member, roles in sorted(self.user_roles.items())
                for role in sorted(roles)
                if full or member == user
            ],
            "AUTHORIZATION_PRIVILEGES": [fact.to_dict() for fact in privileges],
            "AUTHORIZATION_POLICIES": [
                policy.to_dict()
                if full
                else {
                    "name": policy.name,
                    "graph": policy.graph,
                    "actions": sorted(policy.actions),
                    "effect": policy.effect,
                    "enabled": policy.enabled,
                }
                for policy in policies
            ],
            "AUTHORIZATION_POLICY_DEPENDENCIES": [
                {"policy": policy.name, "dependency": dependency}
                for policy in policies
                for dependency in sorted(policy.dependencies)
            ] if full else [],
            "effective_user": user,
            "effective_roles": sorted(effective),
            "catalog_revision": self.revision,
        }
