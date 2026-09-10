from collections import Counter
from itertools import product
import random
import unittest

from authgql.authorization import AuthorizationEngine
from authgql.catalog import AuthorizationCatalog, PolicyDescriptor, PrivilegeFact, Target
from authgql.errors import AuthorizationError, CatalogError
from authgql.executor import SecureExecutor
from authgql.model import PropertyGraph
from authgql.parser import parse_query
from authgql.policy import compile_predicate, PolicyEvaluationContext
from authgql.session import AuthGQLSession
import oracle


def fixture():
    graph = PropertyGraph.from_dict(
        {"name": "g", "nodes": [{"id": "n", "labels": ["N"], "properties": {"flag": 1}}]}
    )
    catalog = AuthorizationCatalog.from_dict(
        {
            "roles": {"reader": [], "senior": ["reader"], "other": []},
            "user_roles": {"alice": ["senior"], "bob": ["other"]},
            "privileges": [
                {
                    "grantee": u,
                    "grantee_kind": "USER",
                    "actions": ["ACCESS", "MATCH", "SET"],
                    "target": {"graph": "g"},
                }
                for u in ["alice", "bob"]
            ]
            + [
                {
                    "grantee": "owner",
                    "grantee_kind": "USER",
                    "actions": ["POLICY REFERENCE"],
                    "target": {"graph": "g"},
                }
            ],
        }
    )
    policy = PolicyDescriptor(
        "visible",
        "g",
        {"TRAVERSE"},
        {"reader"},
        "PERMIT",
        selector={"kind": "NODE", "labels": ["N"]},
        owner="owner",
        using=True,
        grantee_kinds={"reader": "ROLE"},
    )
    return graph, catalog, policy


class RevisionSecurityTests(unittest.TestCase):
    def test_requester_metrics_are_redacted(self):
        g, c, p = fixture()
        c.add_policy(p)
        session = AuthGQLSession({"g": g}, c, "alice")
        result = session.execute("MATCH (n:N) RETURN ID(n) AS id")
        self.assertNotIn("metrics", result)
        with self.assertRaises(AuthorizationError):
            session.execute("SHOW METRICS")

    def test_open_scope_preserves_object_only_access(self):
        g, c, _ = fixture()
        self.assertTrue(AuthorizationEngine(c, g, "bob").permits("TRAVERSE", g.nodes["n"]))

    def test_closed_scope_denies_omitted_broad_grantee(self):
        g, c, p = fixture()
        c.add_policy(p)
        self.assertFalse(AuthorizationEngine(c, g, "bob").permits("TRAVERSE", g.nodes["n"]))
        self.assertTrue(AuthorizationEngine(c, g, "alice").permits("TRAVERSE", g.nodes["n"]))

    def test_activation_reopens_only_after_last_policy_disabled(self):
        g, c, p = fixture()
        c.add_policy(p)
        c.add_policy(
            PolicyDescriptor(
                "also_visible",
                "g",
                {"TRAVERSE"},
                {"reader"},
                "PERMIT",
                owner="owner",
                using=True,
                grantee_kinds={"reader": "ROLE"},
            )
        )
        c.set_policy_enabled("visible", False)
        self.assertFalse(AuthorizationEngine(c, g, "bob").permits("TRAVERSE", g.nodes["n"]))
        c.set_policy_enabled("also_visible", False)
        self.assertTrue(AuthorizationEngine(c, g, "bob").permits("TRAVERSE", g.nodes["n"]))
        c.set_policy_enabled("visible", True)
        self.assertFalse(AuthorizationEngine(c, g, "bob").permits("TRAVERSE", g.nodes["n"]))

    def test_create_policy_requires_explicit_reference_even_for_admin(self):
        g, c, _ = fixture()
        c.add_privilege(
            PrivilegeFact(
                "owner",
                "PERMIT",
                frozenset({"ADMINISTER"}),
                Target("GRAPH", "*"),
                grantee_kind="USER",
            )
        )
        c.privileges = [f for f in c.privileges if "POLICY REFERENCE" not in f.actions]
        session = AuthGQLSession({"g": g}, c, "owner")
        with self.assertRaises(AuthorizationError) as caught:
            session.execute(
                "CREATE AUTHORIZATION POLICY p ON GRAPH g NODES N "
                "FOR TRAVERSE TO USER alice EFFECT PERMIT USING (TRUE)"
            )
        self.assertEqual(caught.exception.code, "42000")
        self.assertEqual(c.policies, [])

    def test_lost_reference_denies_new_snapshot_not_existing_snapshot(self):
        g, c, p = fixture()
        c.add_policy(p)
        session = AuthGQLSession({"g": g}, c, "alice")
        session.execute("BEGIN")
        c.privileges = [f for f in c.privileges if f.grantee != "owner"]
        c.touch()
        query = "MATCH (n:N) RETURN ID(n) AS id"
        self.assertEqual(session.execute(query)["rows"], [{"id": "n"}])
        session.execute("ROLLBACK")
        self.assertEqual(session.execute(query)["rows"], [])

    def test_reference_grant_allows_creation_and_loss_blocks_alter(self):
        g, c, _ = fixture()
        c.add_privilege(
            PrivilegeFact(
                "owner",
                "PERMIT",
                frozenset({"ADMINISTER"}),
                Target("GRAPH", "*"),
                grantee_kind="USER",
            )
        )
        session = AuthGQLSession({"g": g}, c, "owner")
        session.execute(
            "CREATE AUTHORIZATION POLICY p ON GRAPH g NODES N "
            "FOR TRAVERSE TO USER alice EFFECT PERMIT USING (TRUE)"
        )
        self.assertEqual(len(c.policies), 1)
        c.privileges = [f for f in c.privileges if "POLICY REFERENCE" not in f.actions]
        with self.assertRaises(AuthorizationError):
            session.execute("ALTER AUTHORIZATION POLICY p ENABLE")

    def test_explicit_reference_denial_overrides_owner_grant(self):
        g, c, p = fixture()
        c.add_policy(p)
        c.add_privilege(
            PrivilegeFact(
                "owner",
                "DENY",
                frozenset({"POLICY REFERENCE"}),
                Target("GRAPH", "g"),
                grantee_kind="USER",
            )
        )
        self.assertFalse(AuthorizationEngine(c, g, "alice").permits("TRAVERSE", g.nodes["n"]))

    def test_multisupport_cascade_is_conservative(self):
        _, c, _ = fixture()
        graph = Target("GRAPH", "g")
        nodes = Target("NODES", "g", labels=frozenset({"N"}))
        for holder, target, grantor, option in [
            ("alice", graph, "rootA", True),
            ("alice", nodes, "rootD", True),
            ("bob", nodes, "alice", True),
            ("carol", nodes, "bob", False),
        ]:
            c.add_privilege(
                PrivilegeFact(
                    holder, "PERMIT", frozenset({"READ"}), target, option, grantor, "USER"
                )
            )
        c.revoke_privilege("alice", {"READ"}, graph, behavior="CASCADE")
        self.assertTrue(any(f.grantor == "rootD" for f in c.privileges))
        self.assertFalse(any(f.grantor in {"alice", "bob"} for f in c.privileges))

    def test_read_memoization_is_discarded_at_statement_boundary(self):
        from benchmark import ReadMemoization

        g, c, p = fixture()
        c.add_policy(p)
        auth = ReadMemoization(c, g, "alice")
        self.assertTrue(auth.permits("TRAVERSE", g.nodes["n"]))
        self.assertTrue(auth.permits("TRAVERSE", g.nodes["n"]))
        self.assertEqual(auth.hits, 1)
        c.privileges = [f for f in c.privileges if f.grantee != "owner"]
        self.assertFalse(ReadMemoization(c, g, "alice").permits("TRAVERSE", g.nodes["n"]))

    def test_invalid_dependency_remains_closed(self):
        g, c, p = fixture()
        p.dependencies = {"graph:missing"}
        c.add_policy(p)
        self.assertFalse(AuthorizationEngine(c, g, "alice").permits("TRAVERSE", g.nodes["n"]))

    def test_transitive_dependency_owner_loss_denies(self):
        g, c, p = fixture()
        c.add_policy(p)
        p2 = PolicyDescriptor(
            "depends",
            "g",
            {"READ"},
            {"alice"},
            "PERMIT",
            owner="alice",
            dependencies={"policy:visible"},
            grantee_kinds={"alice": "USER"},
        )
        c.add_privilege(
            PrivilegeFact(
                "alice",
                "PERMIT",
                frozenset({"POLICY REFERENCE"}),
                Target("GRAPH", "g"),
                grantee_kind="USER",
            )
        )
        c.add_policy(p2)
        self.assertTrue(AuthorizationEngine(c, g, "alice").permits("READ", g.nodes["n"], "flag"))
        c.privileges = [f for f in c.privileges if f.grantee != "owner"]
        self.assertFalse(AuthorizationEngine(c, g, "alice").permits("READ", g.nodes["n"], "flag"))

    def test_unrelated_invalid_policy_does_not_deny_requester(self):
        g, c, p = fixture()
        c.add_policy(p)
        c.add_policy(
            PolicyDescriptor(
                "bad",
                "g",
                {"TRAVERSE"},
                {"bob"},
                "DENY",
                owner="absent",
                using="unsupported()",
                grantee_kinds={"bob": "USER"},
            )
        )
        self.assertTrue(AuthorizationEngine(c, g, "alice").permits("TRAVERSE", g.nodes["n"]))

    def test_all_three_valued_permit_deny_combinations(self):
        for permit, deny in product([True, False, None], repeat=2):
            g, c, p = fixture()
            p.using = "NULL = 1" if permit is None else permit
            c.add_policy(p)
            c.add_policy(
                PolicyDescriptor(
                    "block",
                    "g",
                    {"TRAVERSE"},
                    {"reader"},
                    "DENY",
                    owner="owner",
                    using="NULL = 1" if deny is None else deny,
                    grantee_kinds={"reader": "ROLE"},
                )
            )
            self.assertEqual(
                AuthorizationEngine(c, g, "alice").permits("TRAVERSE", g.nodes["n"]),
                permit is True and deny is False,
            )

    def test_predicate_whitelist_and_source_bounds(self):
        for source in [
            "evil(RESOURCE)",
            "EXISTS { MATCH (n) DELETE n FINISH }",
            "TRUE" + " " * 8192,
            " AND ".join(["EXISTS { MATCH (n) }"] * 5),
        ]:
            with self.subTest(source=source[:60]), self.assertRaises(CatalogError):
                compile_predicate(source)

    def test_runtime_budget_is_invalid_not_true(self):
        g, c, p = fixture()
        context = PolicyEvaluationContext("alice", set(), resource=g.nodes["n"], remaining_steps=0)
        with self.assertRaises(CatalogError):
            compile_predicate("RESOURCE.flag = 1")(g, context)
        p.using = "EXISTS { MATCH (n) }"
        c.add_policy(p)
        # Exceed the decision budget deterministically without a wall-clock race.
        from unittest.mock import patch

        with patch(
            "authgql.policy.PolicyEvaluationContext.tick", side_effect=CatalogError("budget")
        ):
            self.assertFalse(AuthorizationEngine(c, g, "alice").permits("TRAVERSE", g.nodes["n"]))

    def test_multielement_set_checks_complete_poststate_and_rolls_back(self):
        g, c, _ = fixture()
        g = PropertyGraph.from_dict(
            {
                "name": "g",
                "nodes": [
                    {"id": "n1", "labels": ["N"], "properties": {"flag": 1}},
                    {"id": "n2", "labels": ["N"], "properties": {"flag": 2}},
                ],
            }
        )
        c.add_policy(
            PolicyDescriptor(
                "post",
                "g",
                {"SET"},
                {"alice"},
                "PERMIT",
                owner="owner",
                using=True,
                with_check="NEW_RESOURCE.flag < 3",
                grantee_kinds={"alice": "USER"},
            )
        )
        before = g.to_dict()
        with self.assertRaises(AuthorizationError):
            SecureExecutor(g, c, "alice").execute(parse_query("MATCH (n:N) SET n.flag = 3 FINISH"))
        self.assertEqual(g.to_dict(), before)
        result = SecureExecutor(g, c, "alice").execute(
            parse_query("MATCH (n:N) SET n.flag = 2 FINISH")
        )
        self.assertTrue(result.updated)
        self.assertTrue(all(n.properties["flag"] == 2 for n in g.nodes.values()))


class IndependentOracleTests(unittest.TestCase):
    def test_generated_catalogs_graphs_and_bounded_walks(self):
        # 200 seeds x 3 users x 3 bounds = 1800 complete multiset comparisons.
        for seed in range(200):
            rng = random.Random(seed)
            raw = {"name": "g", "nodes": [], "edges": []}
            for i in range(4):
                props = {} if rng.randrange(4) == 0 else {"flag": rng.choice([0, 1, None])}
                raw["nodes"].append(
                    {
                        "id": str(i),
                        "labels": ["N"] + (["Secret"] if rng.randrange(4) == 0 else []),
                        "properties": props,
                    }
                )
            for i in range(rng.randrange(2, 7)):
                raw["edges"].append(
                    {
                        "id": "e" + str(i),
                        "type": "L",
                        "source": str(rng.randrange(4)),
                        "target": str(rng.randrange(4)),
                        "properties": {"flag": rng.choice([0, 1, None])},
                    }
                )
            depth = rng.randrange(1, 6)
            roles = {"r0": []} | {f"r{i}": [f"r{i - 1}"] for i in range(1, depth)} | {"other": []}
            data = {
                "roles": roles,
                "user_roles": {
                    "alice": [f"r{depth - 1}"],
                    "bob": ["r0", "other"],
                    "carol": ["other"],
                },
                "privileges": [
                    {
                        "grantee": u,
                        "grantee_kind": "USER",
                        "actions": ["ACCESS", "MATCH"],
                        "target": {"graph": "g"},
                    }
                    for u in ["alice", "bob", "carol"]
                ],
                "policies": [],
            }
            data["privileges"].append(
                {
                    "grantee": "owner",
                    "grantee_kind": "USER",
                    "actions": ["POLICY REFERENCE"],
                    "target": {"graph": "g"},
                }
            )
            if rng.choice([False, True]):
                data["privileges"].append(
                    {
                        "grantee": "other",
                        "grantee_kind": "ROLE",
                        "effect": "DENY",
                        "actions": ["TRAVERSE"],
                        "target": {"kind": "NODES", "graph": "g", "labels": ["Secret"]},
                    }
                )
            for i in range(rng.randrange(7)):
                kind = rng.choice(["NODE", "EDGE"])
                spec = rng.choice(
                    [True, False, {"property_compare": {"property": "flag", "value": 1}}]
                )
                data["policies"].append(
                    {
                        "name": f"p{i}",
                        "graph": "g",
                        "owner": "owner",
                        "actions": ["TRAVERSE"],
                        "grantees": ["r0"],
                        "grantee_kinds": {"r0": "ROLE"},
                        "selector": {
                            "kind": kind,
                            **({"labels": ["N"]} if kind == "NODE" else {"edge_types": ["L"]}),
                        },
                        "effect": rng.choice(["PERMIT", "DENY"]),
                        "using": spec,
                        "enabled": rng.choice([True, True, False]),
                    }
                )
            graph, catalog = PropertyGraph.from_dict(raw), AuthorizationCatalog.from_dict(data)
            for user, bound in product(["alice", "bob", "carol"], [1, 2, 3]):
                with self.subTest(seed=seed, user=user, bound=bound):
                    expected = oracle.walks(raw, data, user, bound)
                    query = parse_query(
                        f"MATCH (s:N)-[:L*1..{bound}]->(t:N) RETURN ID(s) AS source, ID(t) AS target"
                    )
                    result = SecureExecutor(graph, catalog, user).execute(query)
                    actual = Counter((r["source"], r["target"]) for r in result.rows)
                    self.assertEqual(actual, expected)
