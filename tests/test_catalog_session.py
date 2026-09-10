from __future__ import annotations

# ruff: noqa: E402

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from authgql.catalog import AuthorizationCatalog, PolicyDescriptor, PrivilegeFact, Target
from authgql.errors import AuthorizationError, CatalogError, ExecutionError, ParseError
from authgql.model import PropertyGraph
from authgql.session import AuthGQLSession


class CatalogAdministrationTests(unittest.TestCase):
    def test_role_hierarchy_is_transitive_and_acyclic(self) -> None:
        catalog = AuthorizationCatalog(roles={"base": set(), "senior": set()})
        catalog.grant_role("base", "senior", "ROLE")
        catalog.grant_role("senior", "alice", "USER")
        self.assertEqual(catalog.effective_roles("alice"), {"base", "senior"})
        with self.assertRaises(CatalogError):
            catalog.grant_role("senior", "base", "ROLE")
        self.assertNotIn("senior", catalog.roles["base"])

    def test_deny_overrides_inherited_permit(self) -> None:
        catalog = AuthorizationCatalog.from_dict(
            {
                "roles": {"reader": [], "restricted": ["reader"], "twin": []},
                "user_roles": {"alice": ["restricted"], "bob": ["twin"]},
                "privileges": [
                    {
                        "grantee": "reader",
                        "effect": "PERMIT",
                        "action": "READ",
                        "target": {"graph": "g"},
                    },
                    {
                        "grantee": "restricted",
                        "effect": "DENY",
                        "action": "READ",
                        "target": {"graph": "g"},
                    },
                    {
                        "grantee": "twin",
                        "grantee_kind": "USER",
                        "effect": "PERMIT",
                        "action": "READ",
                        "target": {"graph": "g"},
                    },
                ],
            }
        )
        self.assertFalse(catalog.object_permitted("alice", "READ", "g"))
        self.assertTrue(catalog.object_permitted("twin", "READ", "g"))
        self.assertFalse(catalog.object_permitted("bob", "READ", "g"))
        catalog.add_policy(
            PolicyDescriptor(
                "user_twin_only",
                "g",
                {"READ"},
                {"twin"},
                "PERMIT",
                grantee_kinds={"twin": "USER"},
            )
        )
        resource = PropertyGraph.from_dict(
            {"name": "g", "nodes": [{"id": "n", "labels": ["N"]}]}
        ).nodes["n"]
        self.assertTrue(catalog.applicable_policies("twin", "READ", "g", resource))
        self.assertFalse(catalog.applicable_policies("bob", "READ", "g", resource))
        self.assertTrue(
            Target("NODES", "g", labels=frozenset({"N"})).covers_target(
                Target(
                    "PROPERTIES",
                    "g",
                    labels=frozenset({"N"}),
                    properties=frozenset({"value"}),
                )
            )
        )
        self.assertTrue(
            Target("NODES", "g", labels=frozenset({"A"})).covers_target(
                Target("NODES", "g", labels=frozenset({"A", "B"})), conjunctive=True
            )
        )
        self.assertFalse(
            Target(
                "PROPERTIES",
                "g",
                labels=frozenset({"A"}),
                properties=frozenset({"x"}),
            ).covers_target(
                Target(
                    "PROPERTIES",
                    "g",
                    labels=frozenset({"A", "B"}),
                    properties=frozenset({"x", "y"}),
                )
            )
        )

    def test_revoke_restrict_and_cascade_follow_delegation(self) -> None:
        session = AuthGQLSession(graphs={"g": PropertyGraph("g")})
        session.execute("CREATE ROLE delegator")
        with self.assertRaises(ParseError):
            session.execute("GRANT READ ON GRAPH g TO ROLE delegator WITH GRANT OPTION")
        with self.assertRaises(ParseError):
            session.execute("GRANT READ ON GRAPH g TO alice WITH GRANT OPTION")
        session.execute("GRANT READ ON GRAPH g TO ROLE delegator")
        session.execute("GRANT delegator TO USER role_only")
        session.execute("SET USER role_only")
        with self.assertRaises(AuthorizationError):
            session.execute("GRANT READ ON GRAPH g TO USER role_child")
        session.execute("SET USER admin")
        session.execute("CREATE ROLE denied_delegate")
        for statement in [
            "GRANT READ ON GRAPH g TO USER alice WITH GRANT OPTION",
            "GRANT SET ON GRAPH g TO USER alice WITH GRANT OPTION",
            "GRANT denied_delegate TO USER alice",
            "DENY READ ON GRAPH g TO ROLE denied_delegate",
            "SET USER alice",
        ]:
            session.execute(statement)
        with self.assertRaises(AuthorizationError):
            session.execute("GRANT READ ON GRAPH g TO USER denied_child")
        for statement in [
            "SET USER admin",
            "REVOKE DENY READ ON GRAPH g FROM ROLE denied_delegate",
            "SET USER alice",
            "GRANT READ ON GRAPH g TO USER bob",
            "GRANT SET ON GRAPH g TO USER carol",
            "SET USER admin",
        ]:
            session.execute(statement)
        with self.assertRaises(CatalogError):
            session.execute("REVOKE GRANT OPTION FOR READ ON GRAPH g FROM USER alice RESTRICT")
        session.execute("REVOKE GRANT OPTION FOR READ ON GRAPH g FROM USER alice CASCADE")
        self.assertTrue(session.catalog.object_permitted("alice", "READ", "g"))
        self.assertFalse(
            session.catalog.delegation_permitted("alice", "READ", Target("GRAPH", "g"))
        )
        self.assertFalse(session.catalog.object_permitted("bob", "READ", "g"))
        self.assertTrue(session.catalog.object_permitted("carol", "SET", "g"))
        self.assertTrue(session.catalog.delegation_permitted("alice", "SET", Target("GRAPH", "g")))

        session.execute("GRANT ADMINISTER ON GRAPH g TO USER scoped_admin")
        session.execute("DENY READ ON GRAPH g TO USER scoped_admin")
        session.execute("SET USER scoped_admin")
        session.execute("GRANT READ ON GRAPH g TO USER scoped_child")
        self.assertTrue(session.catalog.object_permitted("scoped_child", "READ", "g"))
        session.execute("SET USER admin")

        session.execute("GRANT READ ON GRAPH g TO USER dana")
        session.execute("DENY READ ON GRAPH g TO USER dana")
        session.execute("REVOKE READ ON GRAPH g FROM USER dana")
        dana_facts = [fact for fact in session.catalog.privileges if fact.grantee == "dana"]
        self.assertEqual([fact.effect for fact in dana_facts], ["DENY"])
        session.execute("REVOKE DENY READ ON GRAPH g FROM USER dana")
        self.assertFalse(any(fact.grantee == "dana" for fact in session.catalog.privileges))

        session.execute("GRANT READ, SET ON GRAPH g TO USER multi_parent WITH GRANT OPTION")
        session.execute("SET USER multi_parent")
        session.execute("GRANT READ, SET ON GRAPH g TO USER multi_child WITH GRANT OPTION")
        session.execute("SET USER multi_child")
        session.execute("GRANT READ, SET ON GRAPH g TO USER multi_leaf")
        session.execute("SET USER admin")
        with self.assertRaises(CatalogError):
            session.execute(
                "REVOKE GRANT OPTION FOR READ ON GRAPH g FROM USER multi_parent RESTRICT"
            )
        session.execute("REVOKE GRANT OPTION FOR READ ON GRAPH g FROM USER multi_parent CASCADE")
        for user in ["multi_parent", "multi_child", "multi_leaf"]:
            self.assertTrue(session.catalog.object_permitted(user, "SET", "g"))
        self.assertTrue(session.catalog.object_permitted("multi_parent", "READ", "g"))
        self.assertFalse(session.catalog.object_permitted("multi_child", "READ", "g"))
        self.assertFalse(session.catalog.object_permitted("multi_leaf", "READ", "g"))
        self.assertFalse(
            session.catalog.delegation_permitted("multi_parent", "READ", Target("GRAPH", "g"))
        )
        self.assertTrue(
            session.catalog.delegation_permitted("multi_parent", "SET", Target("GRAPH", "g"))
        )
        self.assertTrue(
            session.catalog.delegation_permitted("multi_child", "SET", Target("GRAPH", "g"))
        )
        parent_slices = {
            (fact.actions, fact.grant_option)
            for fact in session.catalog.privileges
            if fact.grantee_kind == "USER" and fact.grantee == "multi_parent"
        }
        self.assertIn((frozenset({"READ"}), False), parent_slices)
        self.assertIn((frozenset({"SET"}), True), parent_slices)
        child_slices = {
            (fact.actions, fact.grant_option)
            for fact in session.catalog.privileges
            if fact.grantee_kind == "USER" and fact.grantee == "multi_child"
        }
        self.assertEqual(child_slices, {(frozenset({"SET"}), True)})

    def test_policy_dependencies_obey_restrict_and_cascade(self) -> None:
        catalog = AuthorizationCatalog(roles={"r": set()})
        catalog.add_policy(PolicyDescriptor("base", "g", {"READ"}, {"r"}, "PERMIT"))
        catalog.add_policy(
            PolicyDescriptor(
                "dependent",
                "g",
                {"READ"},
                {"r"},
                "PERMIT",
                dependencies={"policy:base"},
            )
        )
        with self.assertRaises(CatalogError):
            catalog.drop_policy("base", "RESTRICT")
        removed = catalog.drop_policy("base", "CASCADE")
        self.assertEqual(removed, ["base", "dependent"])

        session = AuthGQLSession(graphs={"g": PropertyGraph("g")})
        session.execute("CREATE ROLE r")
        with self.assertRaises(ParseError):
            session.execute(
                "CREATE AUTHORIZATION POLICY mixed ON GRAPH g FOR READ, SET "
                "TO ROLE r EFFECT PERMIT USING (TRUE)"
            )
        with self.assertRaises(ParseError):
            session.execute(
                "CREATE AUTHORIZATION POLICY old_check ON GRAPH g FOR READ "
                "TO ROLE r EFFECT PERMIT WITH CHECK (TRUE)"
            )
        with self.assertRaises(ParseError):
            session.execute(
                "CREATE AUTHORIZATION POLICY insert_using ON GRAPH g FOR INSERT "
                "TO ROLE r EFFECT PERMIT USING (TRUE)"
            )
        with self.assertRaises(ParseError):
            session.execute(
                "CREATE AUTHORIZATION POLICY update_alias ON GRAPH g FOR UPDATE "
                "TO ROLE r EFFECT PERMIT USING (TRUE)"
            )
        created = session.execute(
            "CREATE AUTHORIZATION POLICY valid ON GRAPH g FOR READ "
            "TO ROLE r EFFECT PERMIT USING (TRUE)"
        )
        self.assertIn("graph:g", created["policy"]["dependencies"])
        with self.assertRaises(CatalogError):
            AuthorizationCatalog.from_dict(
                {
                    "privileges": [
                        {
                            "grantee": "u",
                            "grantee_kind": "USER",
                            "effect": "DENY",
                            "action": "READ",
                            "grant_option": True,
                            "target": {"graph": "g"},
                        }
                    ]
                }
            )

    def test_non_admin_introspection_redacts_predicates_and_dependencies(self) -> None:
        catalog = AuthorizationCatalog.load(ROOT / "examples" / "hospital.catalog.json")
        views = catalog.binding_tables("alice", full=False)
        self.assertTrue(views["AUTHORIZATION_POLICIES"])
        self.assertNotIn("using", views["AUTHORIZATION_POLICIES"][0])
        self.assertEqual(views["AUTHORIZATION_POLICY_DEPENDENCIES"], [])


class TransactionSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = PropertyGraph.from_dict(
            {
                "name": "g",
                "nodes": [{"id": "n", "labels": ["N"], "properties": {"value": 1}}],
            }
        )
        self.catalog = AuthorizationCatalog.from_dict(
            {
                "roles": ["editor"],
                "user_roles": {"alice": ["editor"]},
                "privileges": [
                    {
                        "grantee": "editor",
                        "actions": ["ACCESS", "MATCH", "SET"],
                        "target": {"graph": "g"},
                    }
                ],
            }
        )
        self.session = AuthGQLSession({"g": self.graph}, self.catalog, user="alice", graph_name="g")

    def test_authorization_state_is_fixed_at_transaction_start(self) -> None:
        self.session.execute("START TRANSACTION READ WRITE")
        self.catalog.privileges.clear()
        self.catalog.touch()
        result = self.session.execute("MATCH (n:N) RETURN n.value AS value")
        self.assertEqual(result["rows"], [{"value": 1}])
        self.session.execute("COMMIT")
        with self.assertRaises(AuthorizationError):
            self.session.execute("MATCH (n:N) RETURN n.value AS value")

        admin_session = AuthGQLSession({"g": self.graph.clone()}, user="admin", graph_name="g")
        admin_session.execute("START TRANSACTION READ WRITE")
        admin_session.execute("CREATE ROLE staged_role")
        staged_roles = admin_session.execute("SHOW AUTHORIZATION")["catalog"]["AUTHORIZATION_ROLES"]
        self.assertIn("staged_role", {row["role"] for row in staged_roles})
        admin_session.execute("ROLLBACK")
        committed_roles = admin_session.execute("SHOW AUTHORIZATION")["catalog"][
            "AUTHORIZATION_ROLES"
        ]
        self.assertNotIn("staged_role", {row["role"] for row in committed_roles})

    def test_rollback_discards_staged_graph_update(self) -> None:
        self.session.execute("START TRANSACTION READ WRITE")
        self.session.execute("MATCH (n:N) SET n.value = 2 FINISH")
        self.assertEqual(self.session._active_graph().nodes["n"].properties["value"], 2)
        self.assertEqual(self.graph.nodes["n"].properties["value"], 1)
        self.session.execute("ROLLBACK")
        self.assertEqual(self.graph.nodes["n"].properties["value"], 1)

    def test_authorization_failure_precedes_read_only_mode_failure(self) -> None:
        no_set = self.catalog.clone()
        no_set.privileges = [
            PrivilegeFact("editor", "PERMIT", frozenset({"ACCESS", "MATCH"}), Target("GRAPH", "g"))
        ]
        session = AuthGQLSession({"g": self.graph.clone()}, no_set, "alice", "g")
        session.execute("START TRANSACTION READ ONLY")
        with self.assertRaises(AuthorizationError) as raised:
            session.execute("MATCH (n:N) SET n.value = 2 FINISH")
        self.assertEqual(raised.exception.code, "42000")

    def test_authorized_write_in_read_only_transaction_raises_25g03(self) -> None:
        self.session.execute("START TRANSACTION READ ONLY")
        with self.assertRaises(ExecutionError) as raised:
            self.session.execute("MATCH (n:N) SET n.value = 2 FINISH")
        self.assertEqual(raised.exception.code, "25G03")
        self.assertEqual(self.graph.nodes["n"].properties["value"], 1)


if __name__ == "__main__":
    unittest.main()
