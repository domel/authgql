"""Regression tests for the nine model-completion changes."""

import random
import unittest
from dataclasses import replace
from collections import Counter

from authgql.authorization import AuthorizationEngine
from authgql.catalog import PolicyDescriptor, PrivilegeFact, Target
from authgql.errors import (
    AuthGQLError,
    AuthorizationError,
    ExecutionError,
    ParseError,
    CatalogError,
)
from authgql.executor import SecureExecutor
from authgql.model import PropertyGraph, Edge, Node
from authgql.parser import parse_query
from authgql.policy import compile_predicate
from authgql.session import AuthGQLSession
import oracle


def graph_fixture():
    return PropertyGraph.from_dict(
        {
            "name": "g",
            "nodes": [
                {"id": "same", "labels": ["N"], "properties": {"flag": 1, "secret": "private"}},
                {"id": "b", "labels": ["N"], "properties": {"flag": 1}},
            ],
            "edges": [
                {
                    "id": "same",
                    "labels": ["L", "Extra"],
                    "source": "same",
                    "target": "b",
                    "properties": {"flag": 1, "secret": "private"},
                    "directed": False,
                }
            ],
        }
    )


def catalog_fixture():
    catalog = AuthGQLSession.bootstrap_catalog("owner")
    for user in ["alice", "twin", "member"]:
        catalog.add_privilege(
            PrivilegeFact(
                user,
                "PERMIT",
                frozenset({"ACCESS", "MATCH", "UPDATE"}),
                Target("GRAPH", "g"),
                grantee_kind="USER",
            )
        )
    return catalog


def execute(graph, catalog, text, user="alice", parameters=None):
    return SecureExecutor(graph, catalog, user, parameters).execute(parse_query(text))


def add_policy(catalog, actions, using=True, check=True, name="p", effect="PERMIT", selector=None):
    catalog.add_policy(
        PolicyDescriptor(
            name,
            "g",
            set(actions),
            {"alice"},
            effect,
            selector=selector or {},
            using=using,
            with_check=check,
            owner="owner",
            grantee_kinds={"alice": "USER"},
        )
    )


class ModelCompletionTests(unittest.TestCase):
    def test_staged_policy_dependency_uses_staged_names(self):
        g, c = graph_fixture(), catalog_fixture()
        session = AuthGQLSession({"g": g}, c, "owner")
        session.execute("BEGIN")
        for name, extra in [("parent", ""), ("child", " DEPENDS ON {policy:parent}")]:
            session.execute(
                f"CREATE AUTHORIZATION POLICY {name} ON GRAPH g NODES N "
                f"FOR TRAVERSE TO USER alice EFFECT PERMIT USING (TRUE){extra}"
            )
        self.assertEqual(c.policies, [])
        session.execute("ALTER AUTHORIZATION POLICY child DISABLE")
        session.execute("COMMIT")
        self.assertEqual({p.name for p in c.policies}, {"parent", "child"})

    def test_staged_reference_grant_does_not_change_authority(self):
        g, c = graph_fixture(), catalog_fixture()
        c.privileges = [replace(f, actions=f.actions - {"POLICY REFERENCE"}) for f in c.privileges]
        session = AuthGQLSession({"g": g}, c, "owner")
        session.execute("BEGIN")
        session.execute("GRANT POLICY REFERENCE ON GRAPH g TO USER owner")
        statement = "CREATE AUTHORIZATION POLICY p ON GRAPH g FOR TRAVERSE TO USER alice EFFECT PERMIT USING (TRUE)"
        with self.assertRaises(AuthorizationError):
            session.execute(statement)
        self.assertFalse(session._mutation_catalog().policies)
        session.execute("COMMIT")
        session.execute(statement)
        self.assertEqual(len(c.policies), 1)

    def test_tagged_grantees_roundtrip_and_role_drop(self):
        g, c = graph_fixture(), catalog_fixture()
        session = AuthGQLSession({"g": g}, c, "owner")
        session.execute("CREATE ROLE twin")
        session.execute("GRANT ROLE twin TO USER member")
        session.execute(
            "CREATE AUTHORIZATION POLICY p ON GRAPH g NODES N FOR TRAVERSE "
            "TO USER twin, ROLE twin EFFECT PERMIT USING (TRUE)"
        )
        cloned = c.clone()
        self.assertEqual(cloned.policies[0].principal_keys(), {("USER", "twin"), ("ROLE", "twin")})
        for user in ["twin", "member"]:
            self.assertEqual(len(execute(g, cloned, "MATCH (n:N) RETURN n", user).rows), 2)
        session.execute("DROP ROLE twin CASCADE")
        self.assertEqual(c.policies[0].principal_keys(), {("USER", "twin")})
        self.assertEqual(execute(g, c, "MATCH (n:N) RETURN n", "member").rows, [])
        self.assertEqual(len(execute(g, c, "MATCH (n:N) RETURN n", "twin").rows), 2)

    def test_same_text_ids_set_remove_and_node_edge_equality(self):
        g, c = graph_fixture(), catalog_fixture()
        with self.assertRaises(ExecutionError):
            g.element("same")
        query = "MATCH (n:N)~[e:L]~(m:N) WHERE ID(n) = 'same' "
        result = execute(g, c, query + "SET n.flag = 2, e.flag = 3 RETURN n.flag AS n, e.flag AS e")
        self.assertEqual(result.rows, [{"n": 2, "e": 3}])
        self.assertEqual(result.affected_elements, 2)
        self.assertEqual(execute(g, c, query + "RETURN n = e AS equal").rows, [{"equal": False}])
        result = execute(g, c, query + "REMOVE e.flag RETURN n.flag AS n, e.flag AS e")
        self.assertEqual(result.rows, [{"n": 2, "e": None}])

    def test_same_text_ids_do_not_skip_detach_edge_authorization(self):
        g, c = graph_fixture(), catalog_fixture()
        add_policy(c, {"DELETE"}, using=False, selector={"kind": "EDGE"})
        before = g.to_dict()
        with self.assertRaises(AuthorizationError):
            execute(g, c, "MATCH (n:N) WHERE ID(n) = 'same' DETACH DELETE n FINISH")
        self.assertEqual(g.to_dict(), before)
        c.set_policy_enabled("p", False)
        result = execute(g, c, "MATCH (n:N) WHERE ID(n) = 'same' DETACH DELETE n FINISH")
        self.assertEqual(result.affected_elements, 2)
        self.assertEqual(set(g.nodes), {"b"})
        self.assertEqual(g.edges, {})

    def test_nodetach_requires_kind_qualified_incident_deletions(self):
        g, c = graph_fixture(), catalog_fixture()
        with self.assertRaises(ExecutionError) as caught:
            execute(g, c, "MATCH (n:N) WHERE ID(n) = 'same' DELETE n FINISH")
        self.assertEqual(caught.exception.code, "G1001")
        execute(g, c, "MATCH (n:N)~[e:L]~(m:N) WHERE ID(n) = 'same' DELETE n, e FINISH")
        self.assertEqual(set(g.nodes), {"b"})

    def test_insert_same_text_ids_and_undirected_label_sets(self):
        g, c = PropertyGraph("g"), catalog_fixture()
        identifiers = {"NODE": iter(["same", "b"]), "EDGE": iter(["same"])}
        result = SecureExecutor(
            g, c, "alice", identity_factory=lambda kind: next(identifiers[kind])
        ).execute(parse_query("INSERT (n:N)~[e:L&Extra]~(m:N) RETURN n, e"))
        self.assertEqual(result.affected_elements, 3)
        self.assertEqual(g.edges["same"].labels, {"L", "Extra"})
        self.assertFalse(g.edges["same"].directed)
        self.assertEqual(PropertyGraph.from_dict(g.to_dict()).to_dict(), g.to_dict())

    def test_standard_quantifiers_modes_path_value_and_legacy_agree(self):
        g, c = graph_fixture(), catalog_fixture()
        standard = execute(
            g, c, "MATCH REPEATABLE ELEMENTS p = WALK (n:N)~[:L&Extra]~{0,3}(m:N) RETURN p"
        )
        legacy = execute(g, c, "MATCH p = (n:N)~[:L:Extra*0..3]~(m:N) RETURN p")
        self.assertEqual(standard.rows, legacy.rows)
        self.assertEqual(len(standard.rows), 8)
        for row in standard.rows:
            self.assertEqual(row["p"]["kind"], "path")
            self.assertNotIn("private", str(row))
            for element in row["p"]["elements"]:
                self.assertEqual(set(element), {"kind", "id"})
        composed = execute(g, c, "MATCH p = (a:N)~[:L]~(b:N)~[:L]~(c:N) RETURN p")
        self.assertEqual(len(composed.rows), 2)
        for row in composed.rows:
            elements = row["p"]["elements"]
            self.assertEqual(len(elements), 5)
            self.assertEqual(elements[0], elements[-1])
        self.assertEqual(
            execute(g, c, "MATCH p = (a:N)~[:L]~(b:N), p = (b:N)~[:L]~(a:N) RETURN p").rows, []
        )

    def test_path_bindings_cannot_bypass_denied_intermediate(self):
        g, c = graph_fixture(), catalog_fixture()
        c.add_privilege(
            PrivilegeFact(
                "alice",
                "DENY",
                frozenset({"TRAVERSE"}),
                Target("EDGES", "g", edge_types=frozenset({"Extra"})),
                grantee_kind="USER",
            )
        )
        self.assertEqual(execute(g, c, "MATCH p = (n:N)~[:L]~{1,3}(m:N) RETURN p").rows, [])
        self.assertEqual(len(execute(g, c, "MATCH p = (n:N)~[:L]~{0,0}(m:N) RETURN p").rows), 2)

    def test_directed_undirected_any_direction_and_self_loop_multiplicity(self):
        g, c = graph_fixture(), catalog_fixture()
        for glyph, count in [("-[:L]->", 0), ("<-[:L]-", 0), ("~[:L]~", 2), ("-[:L]-", 2)]:
            self.assertEqual(len(execute(g, c, f"MATCH (n:N){glyph}(m:N) RETURN n").rows), count)
        g.add_edge(Edge("loop", "L", "same", "same", directed=False))
        self.assertEqual(len(execute(g, c, "MATCH (n:N)~[:L]~(m:N) RETURN n").rows), 3)
        self.assertEqual(len(execute(g, c, "MATCH (n:N)-[:L]-(m:N) RETURN n").rows), 3)

    def test_raw_policy_paths_use_new_labels_directions_and_bindings(self):
        g, c = graph_fixture(), catalog_fixture()
        add_policy(
            c,
            {"TRAVERSE"},
            using="EXISTS { MATCH REPEATABLE ELEMENTS p = WALK "
            "(RESOURCE:N)~[:Extra]~{2,2}(m:N) WHERE m = RESOURCE AND p IS NOT NULL }",
            selector={"kind": "NODE"},
        )
        self.assertEqual(len(execute(g, c, "MATCH (n:N) RETURN n").rows), 2)
        g.edges["same"].directed = True
        self.assertEqual(execute(g, c, "MATCH (n:N) RETURN n").rows, [])

    def test_bound_null_and_repeated_edge_bindings_are_not_rebound(self):
        g, c = graph_fixture(), catalog_fixture()
        self.assertEqual(
            execute(
                g, c, "MATCH (n:N) OPTIONAL MATCH (n)-[:Missing]->(m:N) MATCH (m:N) RETURN m"
            ).rows,
            [],
        )
        g.edges["same"].directed = True
        g.add_edge(Edge("back", "L", "b", "same"))
        self.assertEqual(execute(g, c, "MATCH (n:N)-[e:L]->(m:N)-[e:L]->(z:N) RETURN n").rows, [])

    def test_rejected_pattern_syntax_is_not_silently_truncated(self):
        for pattern in [
            "(n:N)~[:L]~{4,2}(m)",
            "(n)~[:L]~{1,13}(m)",
            "(n)~[:L*1..2]~{1,2}(m)",
            "(n:Bad|Other)",
            "(n)~[:L garbage]~(m)",
            "(n:N junk)",
            "DIFFERENT EDGES (n)",
            "p = SHORTEST (n)",
        ]:
            with self.subTest(pattern=pattern), self.assertRaises(ParseError):
                parse_query(f"MATCH {pattern} RETURN n")

    def test_static_names_types_arity_and_path_property_rejection(self):
        g, c = graph_fixture(), catalog_fixture()
        for text in [
            "MATCH (n:N) RETURN missing",
            "MATCH (n:N) WHERE 1 RETURN n",
            "MATCH (n:N) RETURN NOT 1",
            "MATCH (n:N) RETURN 1 = TRUE",
            "MATCH (n:N) RETURN ID()",
            "MATCH (n:N) RETURN unknown(n)",
            "MATCH (n:N) RETURN COUNT(n), n",
            "MATCH (n:N) RETURN COUNT(COUNT(n))",
            "MATCH p = (n:N) RETURN p.secret",
            "MATCH (n)-[n]->(m) RETURN n",
            "MATCH (n:N) RETURN n AS x, n AS x",
            "MATCH p = (n:N) SET p.flag = 1 FINISH",
        ]:
            with self.subTest(text=text), self.assertRaises(AuthGQLError):
                execute(g, c, text)
        for text in ["unbound = 1", "RESOURCE.flag AND 1", "HAS_ROLE()", "PARAM(1)", "1 = TRUE"]:
            with self.subTest(policy=text), self.assertRaises(CatalogError):
                compile_predicate(text)

    def test_parameters_unicode_and_runtime_types(self):
        g, c = graph_fixture(), catalog_fixture()
        g.nodes["same"].properties["city"] = "Białystok"
        result = execute(
            g,
            c,
            "MATCH (n:N {city:$city}) WHERE n.flag = $flag RETURN n.city AS city",
            parameters={"city": "Białystok", "flag": 1},
        )
        self.assertEqual(result.rows, [{"city": "Białystok"}])
        self.assertEqual(
            execute(g, c, "MATCH (n:N {city:'Białystok'}) RETURN n.city AS city").rows, result.rows
        )
        with self.assertRaises(ExecutionError):
            execute(g, c, "MATCH (n:N) WHERE n.flag = $missing RETURN n")
        with self.assertRaises(AuthGQLError):
            execute(g, c, "MATCH (n:N) WHERE n.flag = $flag RETURN n", parameters={"flag": True})
        with self.assertRaises(ExecutionError):
            execute(g, c, "MATCH (n:N) RETURN $injected", parameters={"injected": g.nodes["same"]})
        g.nodes["same"].properties["flag"] = "text"
        with self.assertRaises(ExecutionError):
            execute(g, c, "MATCH (n:N) WHERE n.flag < 2 RETURN n")
        add_policy(c, {"TRAVERSE"}, using="RESOURCE.flag < 2", selector={"kind": "NODE"})
        self.assertEqual(execute(g, c, "MATCH (n:N) RETURN ID(n) AS id").rows, [{"id": "b"}])

    def test_sort_key_guard_runs_on_one_row_and_before_limit(self):
        g, c = graph_fixture(), catalog_fixture()
        add_policy(c, {"READ"}, using=False, selector={"kind": "NODE"})
        for limit in ["", " LIMIT 0"]:
            with self.subTest(limit=limit), self.assertRaises(AuthorizationError):
                execute(
                    g, c, "MATCH (n:N) WHERE ID(n) = 'same' RETURN ID(n) ORDER BY n.secret" + limit
                )
        # A filter that discards every row does not perform the projection read.
        self.assertEqual(execute(g, c, "MATCH (n:N) WHERE FALSE RETURN n.secret").rows, [])

    def test_policy_parameter_errors_deny_and_old_new_values_are_distinct(self):
        g, c = graph_fixture(), catalog_fixture()
        add_policy(
            c,
            {"SET"},
            using="RESOURCE.flag = $old",
            check="RESOURCE.flag = $old AND NEW_RESOURCE.flag = $new",
        )
        before = g.to_dict()
        with self.assertRaises(AuthorizationError):
            execute(g, c, "MATCH (n:N) SET n.flag = 2 FINISH")
        self.assertEqual(g.to_dict(), before)
        execute(g, c, "MATCH (n:N) SET n.flag = 2 FINISH", parameters={"old": 1, "new": 2})
        self.assertTrue(all(n.properties["flag"] == 2 for n in g.nodes.values()))

    def test_reference_option_error_precedes_update_publication(self):
        g, c = graph_fixture(), catalog_fixture()
        before = g.to_dict()
        with self.assertRaises(ExecutionError):
            SecureExecutor(g, c, "alice").execute(
                parse_query("MATCH (n:N) SET n.flag = 2 FINISH"), compare_reference=True
            )
        self.assertEqual(g.to_dict(), before)

    def test_result_limit_does_not_shrink_update_delta_or_count_input(self):
        g, c = graph_fixture(), catalog_fixture()
        result = execute(
            g, c, "MATCH (n:N) SET n.flag = 2 RETURN n.flag AS flag ORDER BY flag LIMIT 1"
        )
        self.assertEqual(result.rows, [{"flag": 2}])
        self.assertEqual(result.affected_elements, 2)
        self.assertTrue(all(n.properties["flag"] == 2 for n in g.nodes.values()))
        self.assertEqual(
            execute(g, c, "MATCH (n:N) RETURN COUNT(n) AS total LIMIT 1").rows, [{"total": 2}]
        )
        self.assertEqual(execute(g, c, "MATCH (n:N) RETURN COUNT(n) AS total LIMIT 0").rows, [])

    def test_order_alias_and_postwrite_projection_failure_roll_back(self):
        g, c = graph_fixture(), catalog_fixture()
        add_policy(c, {"READ"}, using="RESOURCE.flag = 1")
        before = g.to_dict()
        with self.assertRaises(AuthorizationError):
            execute(g, c, "MATCH (n:N) SET n.flag = 2 RETURN n.flag AS flag ORDER BY flag LIMIT 1")
        self.assertEqual(g.to_dict(), before)

    def test_aggregate_order_expression_cannot_suppress_guard(self):
        g, c = graph_fixture(), catalog_fixture()
        add_policy(c, {"READ"}, using=False)
        with self.assertRaises(AuthorizationError):
            execute(g, c, "MATCH (n:N) RETURN COUNT(n) ORDER BY COUNT(n.secret)")

    def test_duplicate_and_out_of_order_clauses_are_rejected(self):
        for text in [
            "MATCH (n) RETURN n RETURN n",
            "MATCH (n) RETURN n WHERE TRUE",
            "MATCH (n) WHERE TRUE MATCH (m) RETURN n",
            "MATCH (n) RETURN n FINISH",
            "MATCH (n) SET n.flag = 1 FINISH junk",
            "MATCH (n) RETURN n LIMIT 1 LIMIT 2",
        ]:
            with self.subTest(text=text), self.assertRaises(ParseError):
                parse_query(text)

    def test_anonymous_nodes_do_not_capture_explicit_or_nested_bindings(self):
        g, c = graph_fixture(), catalog_fixture()
        self.assertEqual(
            execute(g, c, "MATCH (_anon1:N), (:N) RETURN COUNT(*) AS total").rows, [{"total": 4}]
        )
        text = "MATCH (:N {flag: 1}) WHERE EXISTS { MATCH (:N {flag: 2}) } RETURN COUNT(*) AS total"
        g.nodes["b"].properties["flag"] = 2
        self.assertEqual(execute(g, c, text).rows, [{"total": 1}])
        add_policy(
            c,
            {"TRAVERSE"},
            using="EXISTS { MATCH (:N {flag: 1}) WHERE EXISTS { MATCH (:N {flag: 2}) } }",
        )
        self.assertEqual(execute(g, c, "MATCH (n:N) RETURN COUNT(n) AS total").rows, [{"total": 2}])

    def test_nested_exists_labels_do_not_escape_static_scope(self):
        from authgql.analyzer import StaticAnalyzer

        query = parse_query(
            "MATCH (n:N) WHERE EXISTS { MATCH (n:Extra) WHERE n.flag = 1 } RETURN n.secret"
        )
        obligations = StaticAnalyzer(catalog_fixture(), "g", "alice").derive_obligations(query)
        secret = next(o.target for o in obligations if "secret" in o.target.properties)
        flag = next(o.target for o in obligations if "flag" in o.target.properties)
        self.assertEqual(secret.labels, frozenset({"N"}))
        self.assertEqual(flag.labels, frozenset({"N", "Extra"}))

    def test_nested_graph_bindings_cannot_be_stored_as_properties(self):
        g, c = graph_fixture(), catalog_fixture()
        before = g.to_dict()
        with self.assertRaises(ExecutionError):
            execute(g, c, "MATCH (n:N) SET n.flag = LIST(LIST(n)) FINISH")
        self.assertEqual(g.to_dict(), before)

    def test_json_predicates_cannot_ignore_extra_operators(self):
        for spec in [{"all": [], "not": True}, {"all": {}}, {"any": "wrong"}]:
            with self.subTest(spec=spec), self.assertRaises(CatalogError):
                compile_predicate(spec)


class ExtendedOracleTests(unittest.TestCase):
    def test_generated_labelled_oriented_path_values(self):
        # 64 seeds x 4 orientations x 3 upper bounds = 768 multiset comparisons.
        for seed in range(64):
            rng = random.Random(seed)
            raw = {
                "name": "g",
                "nodes": [
                    {
                        "id": str(i),
                        "labels": ["N"],
                        "properties": {"flag": rng.choice([0, 1, None])},
                    }
                    for i in range(3)
                ],
                "edges": [
                    {
                        "id": str(i),
                        "labels": ["L"] + (["Extra"] if rng.randrange(2) else []),
                        "source": str(rng.randrange(3)),
                        "target": str(rng.randrange(3)),
                        "properties": {"flag": rng.choice([0, 1, None])},
                        "directed": bool(rng.randrange(2)),
                    }
                    for i in range(4)
                ],
            }
            c = catalog_fixture()
            add_policy(
                c,
                {"TRAVERSE"},
                using={"property_compare": {"property": "flag", "value": 1}},
                selector={"kind": "EDGE", "edge_types": ["Extra"]},
            )
            data = c.to_dict()
            for direction, glyph in [
                ("out", "-[:L]->"),
                ("in", "<-[:L]-"),
                ("undirected", "~[:L]~"),
                ("both", "-[:L]-"),
            ]:
                for bound in [1, 2, 3]:
                    with self.subTest(seed=seed, direction=direction, bound=bound):
                        expected = oracle.walks(
                            raw,
                            data,
                            "alice",
                            bound,
                            minimum=0,
                            direction=direction,
                            paths=True,
                            labels={"L"},
                        )
                        text = f"MATCH REPEATABLE ELEMENTS p = WALK (s:N){glyph}{{0,{bound}}}(t:N) RETURN p"
                        result = execute(PropertyGraph.from_dict(raw), c, text)
                        actual = Counter(
                            tuple((e["kind"], e["id"]) for e in row["p"]["elements"])
                            for row in result.rows
                        )
                        self.assertEqual(actual, expected)

    def test_generated_complete_poststate_and_atomic_rollback(self):
        # 120 independently computed full graph outcomes, not shared execution.
        for seed in range(120):
            rng = random.Random(seed)
            g, c = graph_fixture(), catalog_fixture()
            for node in g.nodes.values():
                node.properties["flag"] = rng.choice([-1, 0, 1, None, "bad"])
            selected = set(rng.sample(list(g.nodes), rng.randrange(1, 3)))
            value, ceiling = rng.randrange(4), rng.randrange(3)
            add_policy(
                c,
                {"SET"},
                using="RESOURCE.flag >= 0",
                check="NEW_RESOURCE.flag <= $ceiling AND NOT EXISTS { MATCH (m:N) WHERE m.flag <> $value }",
            )
            before = g.to_dict()
            expected_code, expected_graph = oracle.flag_update(before, selected, value, ceiling)
            code = None
            try:
                execute(
                    g,
                    c,
                    "MATCH (n:N) WHERE ID(n) IN ($first, $second) SET n.flag = $value FINISH",
                    parameters={
                        "first": sorted(selected)[0],
                        "second": sorted(selected)[-1],
                        "value": value,
                        "ceiling": ceiling,
                    },
                )
            except AuthGQLError as exc:
                code = exc.code
            with self.subTest(seed=seed):
                self.assertEqual(code, expected_code)
                self.assertEqual(g.to_dict(), expected_graph)

    def test_generated_owner_dependency_snapshot_decisions(self):
        # 80 dependency DAGs, each under an old and a revoked snapshot: 160 outcomes.
        for seed in range(80):
            rng = random.Random(seed)
            g, c = graph_fixture(), catalog_fixture()
            owners, descriptors = {}, {}
            for i in range(5):
                owner = f"o{i}"
                c.add_privilege(
                    PrivilegeFact(
                        owner,
                        "PERMIT",
                        frozenset({"POLICY REFERENCE"}),
                        Target("GRAPH", "g"),
                        grantee_kind="USER",
                    )
                )
                owners[owner] = (True, False)
                deps = {f"p{j}" for j in range(i) if rng.randrange(2)}
                descriptors[f"p{i}"] = {"owner": owner, "depends": deps}
                c.add_policy(
                    PolicyDescriptor(
                        f"p{i}",
                        "g",
                        {"TRAVERSE"},
                        {"alice"},
                        "PERMIT",
                        owner=owner,
                        enabled=i == 4,
                        dependencies={f"policy:{d}" for d in deps},
                        grantee_kinds={"alice": "USER"},
                    )
                )
            frozen = c.clone()
            changed = f"o{rng.randrange(5)}"
            if rng.randrange(2):
                c.add_privilege(
                    PrivilegeFact(
                        changed,
                        "DENY",
                        frozenset({"POLICY REFERENCE"}),
                        Target("GRAPH", "g"),
                        grantee_kind="USER",
                    )
                )
                owners[changed] = (True, True)
            else:
                c.privileges = [f for f in c.privileges if f.grantee != changed]
                owners[changed] = (False, False)
            expected = oracle.owner_references_valid(descriptors, owners, "p4")
            with self.subTest(seed=seed):
                self.assertTrue(
                    AuthorizationEngine(frozen, g, "alice").permits("TRAVERSE", g.nodes["same"])
                )
                self.assertEqual(
                    AuthorizationEngine(c, g, "alice").permits("TRAVERSE", g.nodes["same"]),
                    expected,
                )

    def test_generated_strict_read_and_filter_diagnostics(self):
        # 6 flags x 2 selection results x 2 ordering variants = 24 outcomes.
        for flag in [None, -1, 0, 1, True, "bad"]:
            for selected in [False, True]:
                for sorted_key in [False, True]:
                    g, c = (
                        PropertyGraph("g", {"n": Node("n", {"N"}, {"flag": flag})}),
                        catalog_fixture(),
                    )
                    add_policy(c, {"READ"}, using="RESOURCE.flag = 1")
                    text = f"MATCH (n:N) WHERE {'TRUE' if selected else 'FALSE'} RETURN n.flag AS value"
                    if sorted_key:
                        text += " ORDER BY n.flag"
                    expected_code, expected_rows = oracle.read_outcome(flag, selected, sorted_key)
                    code, rows = None, None
                    try:
                        rows = execute(g, c, text).rows
                    except AuthGQLError as exc:
                        code = exc.code
                    with self.subTest(flag=flag, selected=selected, sort=sorted_key):
                        self.assertEqual(code, expected_code)
                        self.assertEqual(rows, expected_rows)
