"""Focused regressions for the reviewed ISO/IEC 39075 and AuthGQL boundaries."""

import copy
import unittest

from authgql.catalog import AuthorizationCatalog, PolicyDescriptor, PrivilegeFact, Target
from authgql.errors import AuthorizationError, ExecutionError, ParseError, CatalogError
from authgql.executor import SecureExecutor
from authgql.model import Node, Edge, PropertyGraph
from authgql.parser import parse_query
from authgql.session import AuthGQLSession


def fixture():
    graph = PropertyGraph(
        "g",
        {
            "a": Node("a", {"N", "A"}, {"x": 1, "secret": "private"}),
            "b": Node("b", {"N", "B"}, {"x": 2}),
        },
        {"a": Edge("a", "L", "a", "b")},
    )
    return graph, AuthGQLSession.bootstrap_catalog("owner")


def execute(graph, catalog, text, **kwargs):
    return SecureExecutor(graph, catalog, "owner", **kwargs).execute(parse_query(text))


class ISOAlignmentTests(unittest.TestCase):
    def test_delegation_cannot_widen_node_edge_or_property_class_unions(self):
        for kind, field, action, tail in [
            ("NODES", "labels", "TRAVERSE", ""),
            ("EDGES", "edge_types", "TRAVERSE", ""),
            ("PROPERTIES", "labels", "READ", " PROPERTIES {x}"),
        ]:
            with self.subTest(kind=kind):
                target = Target(
                    kind,
                    "g",
                    **{field: frozenset({"A"})},
                    properties=frozenset({"x"}) if kind == "PROPERTIES" else frozenset(),
                )
                catalog = AuthorizationCatalog(
                    privileges=[
                        PrivilegeFact(
                            "alice",
                            "PERMIT",
                            frozenset({action}),
                            target,
                            grant_option=True,
                            grantee_kind="USER",
                        )
                    ]
                )
                catalog.validate()
                session = AuthGQLSession({"g": PropertyGraph("g")}, catalog, "alice")
                before = catalog.to_dict()
                scope = "EDGES" if kind == "EDGES" else "NODES"
                with self.assertRaises(AuthorizationError):
                    session.execute(f"GRANT {action}{tail} ON GRAPH g {scope} {{A,B}} TO USER bob")
                self.assertEqual(catalog.to_dict(), before)
                session.execute(f"GRANT {action}{tail} ON GRAPH g {scope} A TO USER bob")
                if kind == "PROPERTIES":
                    for keys in [frozenset({"x", "secret"}), frozenset()]:
                        self.assertFalse(
                            catalog.delegation_permitted(
                                "alice",
                                "READ",
                                Target("PROPERTIES", "g", labels=frozenset({"A"}), properties=keys),
                            )
                        )

    def test_delegation_narrowing_denial_and_conjunctive_matching(self):
        graph, catalog = fixture()
        session = AuthGQLSession({"g": graph}, catalog, "owner")
        session.execute("GRANT TRAVERSE ON GRAPH g NODES {A,B} TO USER alice WITH GRANT OPTION")
        alice = AuthGQLSession({"g": graph}, catalog, "alice")
        alice.execute("GRANT TRAVERSE ON GRAPH g NODES A TO USER bob")
        self.assertFalse(catalog.object_permitted("bob", "TRAVERSE", "g", graph.nodes["b"]))
        session.execute("DENY TRAVERSE ON GRAPH g NODES A TO USER alice")
        with self.assertRaises(AuthorizationError):
            alice.execute("GRANT TRAVERSE ON GRAPH g NODES {A,B} TO USER bob")
        narrow = AuthorizationCatalog(
            privileges=[
                PrivilegeFact(
                    "owner",
                    "PERMIT",
                    frozenset({"ACCESS"}),
                    Target("GRAPH", "g"),
                    grantee_kind="USER",
                ),
                PrivilegeFact(
                    "owner",
                    "PERMIT",
                    frozenset({"TRAVERSE"}),
                    Target("NODES", "g", labels=frozenset({"A"})),
                    grantee_kind="USER",
                ),
            ]
        )
        narrow.validate()
        self.assertEqual(len(execute(graph, narrow, "MATCH (n:A&N) RETURN n").rows), 1)

    def test_revocation_does_not_confuse_union_overlap_with_support(self):
        graph, catalog = fixture()
        owner = AuthGQLSession({"g": graph}, catalog, "owner")
        owner.execute("GRANT TRAVERSE ON GRAPH g NODES A TO USER alice WITH GRANT OPTION")
        owner.execute("GRANT TRAVERSE ON GRAPH g NODES {A,B} TO USER alice WITH GRANT OPTION")
        AuthGQLSession({"g": graph}, catalog, "alice").execute(
            "GRANT TRAVERSE ON GRAPH g NODES {A,B} TO USER bob"
        )
        owner.execute("REVOKE TRAVERSE ON GRAPH g NODES A FROM USER alice RESTRICT")
        self.assertTrue(catalog.object_permitted("bob", "TRAVERSE", "g", graph.nodes["b"]))
        with self.assertRaises(CatalogError):
            owner.execute("REVOKE TRAVERSE ON GRAPH g NODES {A,B} FROM USER alice RESTRICT")
        owner.execute("REVOKE TRAVERSE ON GRAPH g NODES {A,B} FROM USER alice CASCADE")
        self.assertFalse(catalog.object_permitted("bob", "TRAVERSE", "g", graph.nodes["b"]))

    def test_insert_null_binding_is_not_a_node_constructor(self):
        graph, catalog = fixture()
        before = graph.to_dict()
        result = execute(
            graph, catalog, "MATCH (a:A) OPTIONAL MATCH (b:Missing) INSERT (b) RETURN b"
        )
        self.assertEqual(result.rows, [{"b": None}])
        self.assertEqual(result.affected_elements, 0)
        with self.assertRaises(ExecutionError) as error:
            execute(
                graph,
                catalog,
                "MATCH (a:A) OPTIONAL MATCH (b:Missing) INSERT (z:N), (a)-[e:L]->(b) RETURN e",
            )
        self.assertEqual(error.exception.code, "G1003")
        self.assertEqual(graph.to_dict(), before)

    def test_insert_rejects_bound_or_repeated_decorations_and_edge_names(self):
        statements = [
            "MATCH (a:A) INSERT (a:A) FINISH",
            "MATCH (a:A) INSERT (a {x:99}) FINISH",
            "MATCH (a:A) INSERT (a {}) FINISH",
            "MATCH (a)-[e:L]->(b) INSERT (a)-[e:L]->(b) FINISH",
            "INSERT (a:A)-[e:L]->(b:B), (a)-[e:L]->(b) FINISH",
            "INSERT (a:A), (a:A) FINISH",
            "INSERT (a:A), (a {}) FINISH",
            "INSERT (a:A)-[a:L]->(b:B) FINISH",
        ]
        for text in statements:
            graph, catalog = fixture()
            before = graph.to_dict()
            with self.subTest(text=text), self.assertRaises(ParseError):
                execute(graph, catalog, text)
            self.assertEqual(graph.to_dict(), before)

    def test_insert_reuses_new_nodes_without_graph_wide_insert_authority(self):
        catalog = AuthorizationCatalog(
            privileges=[
                PrivilegeFact(
                    "owner",
                    "PERMIT",
                    frozenset({"ACCESS"}),
                    Target("GRAPH", "g"),
                    grantee_kind="USER",
                ),
                PrivilegeFact(
                    "owner",
                    "PERMIT",
                    frozenset({"INSERT"}),
                    Target("NODES", "g", labels=frozenset({"N"})),
                    grantee_kind="USER",
                ),
                PrivilegeFact(
                    "owner",
                    "PERMIT",
                    frozenset({"INSERT"}),
                    Target("EDGES", "g", edge_types=frozenset({"L"})),
                    grantee_kind="USER",
                ),
            ]
        )
        catalog.validate()
        graph = PropertyGraph("g")
        result = execute(graph, catalog, "INSERT (a:N)-[e:L]->(b:N), (b)-[f:L]->(a) RETURN *")
        self.assertEqual(result.affected_elements, 4)
        self.assertEqual((len(graph.nodes), len(graph.edges)), (2, 2))
        self.assertEqual(set(result.rows[0]), {"a", "b", "e", "f"})

    def test_null_update_targets_are_noops_not_unknown_variables(self):
        for operation in ["SET n.x = 3", "REMOVE n.x", "DELETE n", "DETACH DELETE n"]:
            graph, catalog = fixture()
            before = graph.to_dict()
            with self.subTest(operation=operation):
                result = execute(
                    graph,
                    catalog,
                    f"MATCH (a:A) OPTIONAL MATCH (n:Missing) {operation} RETURN a.x AS x, n",
                )
                self.assertEqual(result.rows, [{"x": 1, "n": None}])
                self.assertEqual(result.affected_elements, 0)
                self.assertEqual(graph.to_dict(), before)
        with self.assertRaises(ParseError):
            execute(graph, catalog, "MATCH (a:A) SET absent.x=3 FINISH")

    def test_null_set_target_still_evaluates_guarded_rhs(self):
        graph, catalog = fixture()
        catalog.add_policy(
            PolicyDescriptor(
                "read",
                "g",
                {"READ"},
                {"owner"},
                "DENY",
                owner="owner",
                using=True,
                grantee_kinds={"owner": "USER"},
            )
        )
        before = graph.to_dict()
        with self.assertRaises(AuthorizationError):
            execute(
                graph, catalog, "MATCH (a:A) OPTIONAL MATCH (n:Missing) SET n.x = a.secret FINISH"
            )
        self.assertEqual(graph.to_dict(), before)
        with self.assertRaises(ExecutionError):
            execute(
                graph, catalog, "MATCH (a:A) OPTIONAL MATCH (n:Missing) SET n.x = $missing FINISH"
            )

    def test_duplicate_set_targets_fail_before_mutation(self):
        graph, catalog = fixture()
        before = graph.to_dict()
        with self.assertRaises(ParseError):
            execute(graph, catalog, "MATCH (a:A) SET a.x=2, a.x=3 FINISH")
        self.assertEqual(graph.to_dict(), before)

    def test_element_id_distinguishes_kinds_and_graphs_without_property_reads(self):
        graph, catalog = fixture()
        catalog.add_policy(
            PolicyDescriptor(
                "read",
                "g",
                {"READ"},
                {"owner"},
                "DENY",
                owner="owner",
                using=True,
                grantee_kinds={"owner": "USER"},
            )
        )
        result = execute(
            graph,
            catalog,
            "MATCH (a:A)-[e:L]->(b) "
            "RETURN a=e AS refs, ELEMENT_ID(a)=ELEMENT_ID(e) AS ids, "
            "ELEMENT_ID(a)=ELEMENT_ID(a) AS stable, ID(a)=ID(e) AS legacy",
        )
        self.assertEqual(
            result.rows, [{"refs": False, "ids": False, "stable": True, "legacy": True}]
        )
        first = execute(graph, catalog, "MATCH (a:A) RETURN ELEMENT_ID(a) AS id").rows[0]["id"]
        other = graph.clone()
        other.name = "other"
        second = execute(
            other,
            AuthGQLSession.bootstrap_catalog("owner"),
            "MATCH (a:A) RETURN ELEMENT_ID(a) AS id",
        ).rows[0]["id"]
        self.assertNotEqual(first, second)
        self.assertEqual(
            execute(
                graph, catalog, "MATCH (a:A) OPTIONAL MATCH (m:Missing) RETURN ELEMENT_ID(m) AS id"
            ).rows,
            [{"id": None}],
        )

    def test_id_property_is_ordinary_data_and_fixture_ids_are_separate(self):
        graph, catalog = PropertyGraph("g"), AuthGQLSession.bootstrap_catalog("owner")
        ids = {"NODE": iter(["a", "b"]), "EDGE": iter(["a"])}
        result = execute(
            graph,
            catalog,
            "INSERT (a:N {_id:'user-value'})-[e:L {_id:'edge-value'}]->(b:N) "
            "RETURN a._id AS node_value, e._id AS edge_value",
            identity_factory=lambda kind: next(ids[kind]),
        )
        self.assertEqual(result.rows, [{"node_value": "user-value", "edge_value": "edge-value"}])
        self.assertEqual(graph.nodes["a"].properties["_id"], "user-value")
        self.assertEqual(graph.edges["a"].properties["_id"], "edge-value")
        generated, catalog = PropertyGraph("g"), AuthGQLSession.bootstrap_catalog("owner")
        execute(generated, catalog, "INSERT (n:N {_id:'x'}) FINISH")
        self.assertNotIn("x", generated.nodes)
        self.assertEqual(next(iter(generated.nodes.values())).properties["_id"], "x")

    def test_return_star_projects_only_named_opaque_bindings(self):
        graph, catalog = fixture()
        catalog.add_policy(
            PolicyDescriptor(
                "read",
                "g",
                {"READ"},
                {"owner"},
                "DENY",
                owner="owner",
                using=True,
                grantee_kinds={"owner": "USER"},
            )
        )
        result = execute(
            graph, catalog, "MATCH p = (a:A)-[e:L]->() OPTIONAL MATCH (m:Missing) RETURN *"
        )
        self.assertEqual(set(result.rows[0]), {"a", "e", "p", "m"})
        self.assertEqual(result.columns, ["a", "e", "m", "p"])
        self.assertNotIn("private", str(result.rows))
        self.assertNotIn("labels", str(result.rows))
        self.assertIsNone(result.rows[0]["m"])
        for text in [
            "MATCH (n) RETURN * AS x",
            "MATCH (n) RETURN *, n",
            "MATCH () RETURN *",
            "INSERT (:N) RETURN *",
        ]:
            with self.assertRaises(ParseError):
                execute(graph, catalog, text)

    def test_crosscheck_preserves_raw_context_for_positive_and_negative_exists(self):
        for predicate in ["EXISTS { MATCH (h:Hidden) }", "NOT EXISTS { MATCH (h:Hidden) }"]:
            graph, catalog = fixture()
            graph.add_node(Node("hidden", {"Hidden"}))
            catalog.add_policy(
                PolicyDescriptor(
                    "hide",
                    "g",
                    {"TRAVERSE"},
                    {"owner"},
                    "DENY",
                    selector={"kind": "NODE", "labels": ["Hidden"]},
                    owner="owner",
                    using=True,
                    grantee_kinds={"owner": "USER"},
                )
            )
            catalog.add_policy(
                PolicyDescriptor(
                    "visible",
                    "g",
                    {"TRAVERSE"},
                    {"owner"},
                    "PERMIT",
                    selector={"kind": "NODE", "labels": ["N"]},
                    owner="owner",
                    using=predicate,
                    grantee_kinds={"owner": "USER"},
                )
            )
            query = parse_query("MATCH (n:N) RETURN n")
            expected = SecureExecutor(graph, catalog, "owner").execute(query).rows
            with self.subTest(predicate=predicate):
                result = SecureExecutor(graph, catalog, "owner").execute(
                    query, compare_reference=True
                )
                self.assertTrue(result.reference_equal)
                self.assertEqual(result.rows, expected)

    def test_crosscheck_preserves_raw_context_for_property_guards(self):
        graph, catalog = fixture()
        graph.add_node(Node("hidden", {"Hidden"}))
        catalog.add_policy(
            PolicyDescriptor(
                "hide",
                "g",
                {"TRAVERSE"},
                {"owner"},
                "DENY",
                selector={"kind": "NODE", "labels": ["Hidden"]},
                owner="owner",
                using=True,
                grantee_kinds={"owner": "USER"},
            )
        )
        catalog.add_policy(
            PolicyDescriptor(
                "read",
                "g",
                {"READ"},
                {"owner"},
                "PERMIT",
                owner="owner",
                using="EXISTS { MATCH (h:Hidden) }",
                grantee_kinds={"owner": "USER"},
            )
        )
        result = SecureExecutor(graph, catalog, "owner").execute(
            parse_query("MATCH (n:N) RETURN n.x AS x ORDER BY x"), compare_reference=True
        )
        self.assertTrue(result.reference_equal)
        self.assertEqual(result.rows, [{"x": 1}, {"x": 2}])

    def test_buffered_result_failure_leaves_no_published_write(self):
        graph, catalog = fixture()
        catalog.add_policy(
            PolicyDescriptor(
                "read",
                "g",
                {"READ"},
                {"owner"},
                "PERMIT",
                owner="owner",
                using="RESOURCE.x = 1",
                grantee_kinds={"owner": "USER"},
            )
        )
        session = AuthGQLSession({"g": graph}, catalog, "owner")
        before = copy.deepcopy(graph.to_dict())
        with self.assertRaises(AuthorizationError):
            session.execute("MATCH (n:N) SET n.y=3 RETURN n.x")
        self.assertEqual(graph.to_dict(), before)
