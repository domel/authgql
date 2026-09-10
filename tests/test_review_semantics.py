"""Regressions for guarded READ, retained policy scopes and explicit write scope."""

import unittest

from authgql.authorization import AuthorizationEngine
from authgql.analyzer import StaticAnalyzer
from authgql.catalog import AuthorizationCatalog, PolicyDescriptor, PrivilegeFact, Target
from authgql.errors import AuthorizationError, CatalogError
from authgql.executor import SecureExecutor
from authgql.model import Node, Edge, PropertyGraph
from authgql.parser import parse_query
from authgql.session import AuthGQLSession


def fixture():
    graph = PropertyGraph("g", {"n": Node("n", {"N"}, {"name": "Public", "secret": "Private"})})
    catalog = AuthGQLSession.bootstrap_catalog("owner")
    for user in ["reader", "member"]:
        catalog.add_privilege(
            PrivilegeFact(
                user,
                "PERMIT",
                frozenset({"ACCESS", "MATCH", "INSERT"}),
                Target("GRAPH", "g"),
                grantee_kind="USER",
            )
        )
    return graph, catalog


def run(graph, catalog, query, user="reader", **options):
    return SecureExecutor(graph, catalog, user).execute(parse_query(query), **options)


def protect_role(graph, catalog):
    admin = AuthGQLSession({"g": graph}, catalog, "owner")
    admin.execute("CREATE ROLE r")
    admin.execute("GRANT ROLE r TO USER member")
    admin.execute(
        "CREATE AUTHORIZATION POLICY p ON GRAPH g NODES N "
        "FOR READ TO ROLE r EFFECT PERMIT USING (TRUE)"
    )
    return admin


class ReviewSemanticsTests(unittest.TestCase):
    def test_poststate_policy_preserves_old_node_bindings(self):
        patterns = [
            "(RESOURCE:N)<-[:R]-(a:A)",
            "(a:A)-[:R]->(RESOURCE:N)",
            "(n:N)-[:R]->{0,0}(RESOURCE:N)",
            "(RESOURCE:N {flag:0})<-[:R]-(a:A)",
            "(a:A)-[:R]->(RESOURCE:N {flag:0})",
            "(n:N)-[:R]->{0,0}(RESOURCE:N {flag:0})",
        ]
        for pattern in patterns:
            for old_flag in (0, 1):
                with self.subTest(pattern=pattern, old_flag=old_flag):
                    graph = PropertyGraph(
                        "g",
                        {"n": Node("n", {"N"}, {"flag": 0}), "a": Node("a", {"A"})},
                        {"e": Edge("e", "R", "a", "n")},
                    )
                    catalog = AuthGQLSession.bootstrap_catalog("owner")
                    admin = AuthGQLSession({"g": graph}, catalog, "owner")
                    admin.execute("GRANT ACCESS, MATCH, SET ON GRAPH g TO USER reader")
                    admin.execute(
                        "CREATE AUTHORIZATION POLICY p ON GRAPH g NODES N "
                        "FOR SET TO USER reader EFFECT PERMIT USING (TRUE) WITH CHECK ("
                        f"EXISTS {{ MATCH {pattern} WHERE RESOURCE.flag = {old_flag} "
                        "AND NEW_RESOURCE.flag = 1 })"
                    )
                    query = "MATCH (n:N) SET n.flag = 1 FINISH"
                    if old_flag == 0:
                        run(graph, catalog, query)
                        self.assertEqual(graph.nodes["n"].properties["flag"], 1)
                    else:
                        before = graph.to_dict()
                        with self.assertRaises(AuthorizationError):
                            run(graph, catalog, query)
                        self.assertEqual(graph.to_dict(), before)

    def test_poststate_policy_preserves_old_edge_bindings(self):
        for decoration in ("", " {flag:0}"):
            for old_flag in (0, 1):
                with self.subTest(decoration=decoration, old_flag=old_flag):
                    graph = PropertyGraph(
                        "g",
                        {"a": Node("a", {"A"}), "b": Node("b", {"B"})},
                        {"e": Edge("e", "R", "a", "b", {"flag": 0})},
                    )
                    catalog = AuthGQLSession.bootstrap_catalog("owner")
                    admin = AuthGQLSession({"g": graph}, catalog, "owner")
                    admin.execute("GRANT ACCESS, MATCH, SET ON GRAPH g TO USER reader")
                    admin.execute(
                        "CREATE AUTHORIZATION POLICY p ON GRAPH g EDGES R "
                        "FOR SET TO USER reader EFFECT PERMIT USING (TRUE) WITH CHECK ("
                        f"EXISTS {{ MATCH (a:A)-[RESOURCE:R{decoration}]->(b:B) "
                        f"WHERE RESOURCE.flag = {old_flag} AND NEW_RESOURCE.flag = 1 }})"
                    )
                    query = "MATCH (a:A)-[e:R]->(b:B) SET e.flag = 1 FINISH"
                    if old_flag == 0:
                        run(graph, catalog, query)
                        self.assertEqual(graph.edges["e"].properties["flag"], 1)
                    else:
                        before = graph.to_dict()
                        with self.assertRaises(AuthorizationError):
                            run(graph, catalog, query)
                        self.assertEqual(graph.to_dict(), before)

    def test_drop_role_restrict_checks_both_inheritance_directions(self):
        graph, catalog = fixture()
        original_roles = {role: set(parents) for role, parents in catalog.roles.items()}
        admin = AuthGQLSession({"g": graph}, catalog, "owner")
        admin.execute("CREATE ROLE parent")
        admin.execute("CREATE ROLE child")
        admin.execute("GRANT ROLE parent TO ROLE child")
        before = catalog.to_dict()
        for role in ("parent", "child"):
            for suffix in ("", " RESTRICT"):
                with self.subTest(role=role, suffix=suffix):
                    with self.assertRaises(CatalogError):
                        admin.execute(f"DROP ROLE {role}{suffix}")
                    self.assertEqual(catalog.to_dict(), before)
        admin.execute("DROP ROLE child CASCADE")
        self.assertEqual(catalog.roles, original_roles | {"parent": set()})
        admin.execute("DROP ROLE parent RESTRICT")
        self.assertEqual(catalog.roles, original_roles)

    def test_conservative_profile_requires_static_obligations(self):
        graph, _ = fixture()
        catalog = AuthGQLSession.bootstrap_catalog("owner")
        admin = AuthGQLSession({"g": graph}, catalog, "owner")
        admin.execute("GRANT ACCESS, TRAVERSE ON GRAPH g TO USER reader")
        query = "MATCH (n:N) WHERE FALSE RETURN n.secret"
        with self.assertRaises(AuthorizationError):
            run(graph, catalog, query)
        admin.execute("GRANT READ PROPERTIES {secret} ON GRAPH g NODES N TO USER reader")
        self.assertEqual(run(graph, catalog, query).rows, [])

    def test_insert_constructor_requires_only_initialization_obligations(self):
        graph = PropertyGraph("g")
        catalog = AuthGQLSession.bootstrap_catalog("owner")
        admin = AuthGQLSession({"g": graph}, catalog, "owner")
        admin.execute("GRANT ACCESS, INSERT ON GRAPH g TO USER reader")
        query = parse_query("INSERT (n:N {flag:1}) FINISH")
        obligations = StaticAnalyzer(catalog, "g", "reader").derive_obligations(query)
        self.assertEqual({o.action for o in obligations}, {"ACCESS", "INSERT"})
        result = SecureExecutor(graph, catalog, "reader").execute(query)
        self.assertEqual(result.affected_elements, 1)
        self.assertEqual(next(iter(graph.nodes.values())).properties, {"flag": 1})

    def test_topology_view_does_not_replace_guarded_read(self):
        graph, catalog = fixture()
        catalog.add_policy(
            PolicyDescriptor(
                "read",
                "g",
                {"READ"},
                {"reader"},
                "DENY",
                owner="owner",
                using=True,
                grantee_kinds={"reader": "USER"},
            )
        )
        self.assertEqual(
            len(run(graph, catalog, "MATCH (n:N) RETURN n", compare_reference=True).rows), 1
        )
        for crosscheck in [False, True]:
            with self.assertRaises(AuthorizationError):
                run(graph, catalog, "MATCH (n:N) RETURN n.secret", compare_reference=crosscheck)
        # An ordinary unguarded property lookup would still disclose this value.
        self.assertEqual(graph.nodes["n"].properties["secret"], "Private")

    def test_patient_base_permit_and_sealed_exception_cover_both_actions(self):
        graph = PropertyGraph(
            "hospital",
            {
                "u": Node("u", {"User"}, {"login": "alice"}),
                "h": Node("h", {"Hospital"}),
                "ordinary": Node(
                    "ordinary", {"Patient"}, {"classification": "ordinary", "age": 42}
                ),
                "sealed": Node("sealed", {"Patient"}, {"classification": "sealed", "age": 50}),
            },
            {
                "work": Edge("work", "WORKS_AT", "u", "h"),
                "t1": Edge("t1", "TREATED_AT", "ordinary", "h"),
                "t2": Edge("t2", "TREATED_AT", "sealed", "h"),
            },
        )
        catalog = AuthGQLSession.bootstrap_catalog("owner")
        admin = AuthGQLSession({"hospital": graph}, catalog, "owner")
        for statement in [
            "CREATE ROLE clinician",
            "GRANT ROLE clinician TO USER alice",
            "GRANT ACCESS, MATCH ON GRAPH hospital TO ROLE clinician",
            "CREATE AUTHORIZATION POLICY patient_visibility ON GRAPH hospital NODES Patient "
            "FOR TRAVERSE, READ TO ROLE clinician EFFECT PERMIT USING ("
            "EXISTS { MATCH (u:User {login: SESSION_USER})-[:WORKS_AT]->(h:Hospital) "
            "MATCH (RESOURCE:Patient)-[:TREATED_AT]->(h) })",
            "CREATE AUTHORIZATION POLICY sealed_patient_records ON GRAPH hospital NODES Patient "
            "FOR TRAVERSE, READ TO ROLE clinician EFFECT DENY USING (RESOURCE.classification = 'sealed' "
            "AND NOT EXISTS { MATCH (u:User {login: SESSION_USER})-[:HAS_CLEARANCE]->"
            "(:Clearance {level: 'sealed'}) })",
        ]:
            admin.execute(statement)
        self.assertEqual(
            run(
                graph,
                catalog,
                "MATCH (p:Patient) RETURN p.age AS age",
                user="alice",
                compare_reference=True,
            ).rows,
            [{"age": 42}],
        )

    def test_drop_last_role_retains_closed_scope_and_roundtrips(self):
        graph, catalog = fixture()
        admin = protect_role(graph, catalog)
        with self.assertRaises(AuthorizationError):
            run(graph, catalog, "MATCH (n:N) RETURN n.name")
        with self.assertRaises(CatalogError):
            admin.execute("DROP ROLE r RESTRICT")
        admin.execute("DROP ROLE r CASCADE")
        policy = catalog.policies[0]
        self.assertEqual(policy.principal_keys(), set())
        self.assertTrue(policy.enabled)
        self.assertEqual(policy.actions, {"READ"})
        for restored in [catalog, AuthorizationCatalog.from_dict(catalog.to_dict())]:
            with self.assertRaises(AuthorizationError):
                run(graph, restored, "MATCH (n:N) RETURN n.name")
        admin.execute("CREATE ROLE r")
        admin.execute("GRANT ROLE r TO USER member")
        with self.assertRaises(AuthorizationError):
            run(graph, catalog, "MATCH (n:N) RETURN n.name", user="member")
        admin.execute("DROP AUTHORIZATION POLICY p")
        self.assertEqual(
            run(graph, catalog, "MATCH (n:N) RETURN n.name AS name").rows, [{"name": "Public"}]
        )

    def test_empty_policy_retains_dependencies_and_snapshot_authority(self):
        graph, catalog = fixture()
        admin = protect_role(graph, catalog)
        admin.execute(
            "CREATE AUTHORIZATION POLICY child ON GRAPH g NODES N "
            "FOR READ TO ROLE r EFFECT PERMIT USING (TRUE) DEPENDS ON {policy:p}"
        )
        member = AuthGQLSession({"g": graph}, catalog, "member")
        member.execute("BEGIN")
        admin.execute("DROP ROLE r CASCADE")
        self.assertEqual({p.name for p in catalog.policies}, {"p", "child"})
        child = next(p for p in catalog.policies if p.name == "child")
        self.assertIn("policy:p", child.dependencies)
        self.assertTrue(member.execute("MATCH (n:N) RETURN n.name")["rows"])
        member.execute("ROLLBACK")
        with self.assertRaises(AuthorizationError):
            member.execute("MATCH (n:N) RETURN n.name")
        with self.assertRaises(CatalogError):
            admin.execute("DROP AUTHORIZATION POLICY p RESTRICT")
        admin.execute("DROP AUTHORIZATION POLICY p CASCADE")
        self.assertEqual(catalog.policies, [])

    def test_policy_creation_still_requires_a_grantee(self):
        graph, catalog = fixture()
        with self.assertRaises(CatalogError):
            catalog.add_policy(PolicyDescriptor("p", "g", {"READ"}, set(), "PERMIT", owner="owner"))
        self.assertEqual(catalog.policies, [])

    def test_insert_initial_properties_are_not_subsequent_set(self):
        graph, catalog = fixture()
        catalog.add_privilege(
            PrivilegeFact(
                "reader",
                "DENY",
                frozenset({"SET"}),
                Target(
                    "PROPERTIES",
                    "g",
                    labels=frozenset({"N"}),
                    properties=frozenset({"classification"}),
                ),
                grantee_kind="USER",
            )
        )
        run(graph, catalog, "INSERT (:N {classification:'sealed'}) FINISH")
        self.assertEqual(len(graph.nodes), 2)
        before = graph.to_dict()
        with self.assertRaises(AuthorizationError):
            run(graph, catalog, "MATCH (n:N) SET n.classification='sealed' FINISH")
        catalog.add_policy(
            PolicyDescriptor(
                "insert",
                "g",
                {"INSERT"},
                {"reader"},
                "PERMIT",
                owner="owner",
                with_check="NEW_RESOURCE.classification <> 'sealed'",
                grantee_kinds={"reader": "USER"},
            )
        )
        with self.assertRaises(AuthorizationError):
            run(graph, catalog, "INSERT (:N {classification:'sealed'}) FINISH")
        self.assertEqual(graph.to_dict(), before)
        run(graph, catalog, "INSERT (:N {classification:'ordinary'}) FINISH")
        self.assertEqual(len(graph.nodes), 3)

    def test_element_read_policy_closes_all_keys(self):
        graph, catalog = fixture()
        catalog.add_policy(
            PolicyDescriptor(
                "read",
                "g",
                {"READ"},
                {"reader"},
                "PERMIT",
                owner="owner",
                using=False,
                selector={"kind": "NODE", "labels": ["N"]},
                grantee_kinds={"reader": "USER"},
            )
        )
        engine = AuthorizationEngine(catalog, graph, "reader")
        for key in ["name", "secret", "absent"]:
            with self.assertRaises(AuthorizationError):
                engine.check("READ", graph.nodes["n"], key)

    def test_reference_plan_filters_before_projection_read(self):
        graph, catalog = fixture()
        catalog.add_policy(
            PolicyDescriptor(
                "read",
                "g",
                {"READ"},
                {"reader"},
                "DENY",
                owner="owner",
                using=True,
                grantee_kinds={"reader": "USER"},
            )
        )
        self.assertEqual(
            run(
                graph,
                catalog,
                "MATCH (n:N) WHERE n:Missing RETURN n.secret",
                compare_reference=True,
            ).rows,
            [],
        )
        with self.assertRaises(AuthorizationError):
            run(graph, catalog, "MATCH (n:N) WHERE n.secret = 'Private' RETURN n")
