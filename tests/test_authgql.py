from __future__ import annotations

# ruff: noqa: E402

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from authgql.analyzer import StaticAnalyzer
from authgql.catalog import AuthorizationCatalog, PolicyDescriptor
from authgql.errors import AuthorizationError, CatalogError, ExecutionError
from authgql.executor import SecureExecutor
from authgql.model import PropertyGraph
from authgql.parser import parse_query
from authgql.policy import PolicyEvaluationContext, compile_predicate


EXAMPLES = ROOT / "examples"


class HospitalSecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = PropertyGraph.load(EXAMPLES / "hospital.graph.json")
        self.catalog = AuthorizationCatalog.load(EXAMPLES / "hospital.catalog.json")

    def run_query(self, source: str, user: str = "alice", compare: bool = False):
        return SecureExecutor(self.graph, self.catalog, user).execute(
            parse_query(source), compare_reference=compare
        )

    def test_guarded_path_hides_remote_and_sealed_patients(self) -> None:
        result = self.run_query(
            "MATCH (p:Patient)-[:TREATED_AT]->(h:Hospital) "
            "OPTIONAL MATCH (p)-[:HAS_DIAGNOSIS]->(d:Diagnosis) "
            "RETURN p.age AS age, d.code AS code",
            compare=True,
        )
        self.assertEqual(result.rows, [{"age": 42, "code": "D-A"}])
        self.assertTrue(result.reference_equal)
        self.assertGreaterEqual(result.metrics.denied_before_enqueue, 2)

    def test_researcher_cannot_dereference_identifying_property(self) -> None:
        with self.assertRaises(AuthorizationError) as raised:
            self.run_query(
                "MATCH (p:Patient)-[:HAS_DIAGNOSIS]->(d:Diagnosis) "
                "RETURN p.name AS name",
                user="rita",
            )
        self.assertEqual(raised.exception.code, "42000")

    def test_whole_element_projection_is_an_opaque_reference(self) -> None:
        result = self.run_query(
            "MATCH (p:Patient) RETURN p AS patient",
            user="rita",
        )
        self.assertEqual(
            result.rows,
            [{"patient": {"id": "patient_visible", "kind": "node"}}],
        )
        self.assertNotIn("properties", result.rows[0]["patient"])

    def test_optional_match_null_extends_after_authorization(self) -> None:
        result = self.run_query(
            "MATCH (h:Hospital {code:'H2'}) "
            "OPTIONAL MATCH (p:Patient)-[:TREATED_AT]->(h) "
            "RETURN p.age AS age"
        )
        self.assertEqual(result.rows, [{"age": None}])
        unknown_filter = self.run_query(
            "MATCH (h:Hospital {code:'H2'}) "
            "OPTIONAL MATCH (p:Patient)-[:TREATED_AT]->(h) "
            "WHERE NOT p.age = 1 RETURN h.code AS hospital"
        )
        self.assertEqual(unknown_filter.rows, [])

    def test_exists_cannot_observe_hidden_patient(self) -> None:
        result = self.run_query(
            "MATCH (h:Hospital {code:'H2'}) "
            "WHERE EXISTS { MATCH (p:Patient)-[:TREATED_AT]->(h) } "
            "RETURN h.code AS hospital"
        )
        self.assertEqual(result.rows, [])

    def test_aggregate_counts_only_authorized_bindings(self) -> None:
        result = self.run_query("MATCH (p:Patient) RETURN COUNT(p) AS patients")
        self.assertEqual(result.rows, [{"patients": 1}])

    def test_denied_sort_key_fails_before_sorting(self) -> None:
        graph = PropertyGraph.from_dict(
            {
                "name": "g",
                "nodes": [
                    {"id": "open", "labels": ["N"], "properties": {"rank": 1}},
                    {
                        "id": "secret",
                        "labels": ["N", "Secret"],
                        "properties": {"rank": 2, "secret": "TOP-SECRET"},
                    },
                ],
            }
        )
        catalog = AuthorizationCatalog.from_dict(
            {
                "roles": ["reader"],
                "user_roles": {"u": ["reader"]},
                "privileges": [
                    {
                        "grantee": "reader",
                        "actions": ["ACCESS", "MATCH"],
                        "target": {"graph": "g"},
                    }
                ],
                "policies": [
                    {
                        "name": "deny_secret_sort_key",
                        "graph": "g",
                        "actions": ["READ"],
                        "grantees": ["reader"],
                        "effect": "DENY",
                        "selector": {"kind": "NODE", "labels": ["Secret"]},
                        "using": True,
                    }
                ],
            }
        )
        executor = SecureExecutor(graph, catalog, "u")
        with self.assertRaises(AuthorizationError):
            executor.execute(
                parse_query("MATCH (p:N) RETURN ID(p) AS id ORDER BY p.rank")
            )
        self.assertGreater(executor.metrics.policy_denials, 0)

        catalog.policies[0].using = "RESOURCE.secret"
        with self.assertRaises(AuthorizationError) as fault:
            SecureExecutor(graph, catalog, "u").execute(
                parse_query("MATCH (p:N) RETURN p.rank AS rank")
            )
        self.assertEqual(str(fault.exception), "Authorization policy evaluation failed")
        self.assertNotIn("TOP-SECRET", str(fault.exception))

    def test_with_check_rejects_and_rolls_back_set(self) -> None:
        update_policy = next(
            policy
            for policy in self.catalog.policies
            if policy.name == "local_patient_update"
        )
        update_policy.with_check = (
            "RESOURCE.classification = 'normal' AND "
            "NEW_RESOURCE.classification = 'restricted'"
        )
        result = self.run_query(
            "MATCH (p:Patient) WHERE p.age = 42 "
            "SET p.classification = 'restricted' FINISH"
        )
        self.assertTrue(result.updated)
        self.assertEqual(
            self.graph.nodes["patient_visible"].properties["classification"],
            "restricted",
        )

        update_policy.with_check = (
            "RESOURCE.classification = 'restricted' AND "
            "NEW_RESOURCE.classification <> 'sealed'"
        )
        before = self.graph.to_dict()
        with self.assertRaises(AuthorizationError) as raised:
            self.run_query(
                "MATCH (p:Patient) WHERE p.age = 42 "
                "SET p.classification = 'sealed' FINISH"
            )
        self.assertEqual(raised.exception.code, "42000")
        self.assertEqual(self.graph.to_dict(), before)

        update_policy.with_check = "NEW_RESOURCE.classification < 1"
        with self.assertRaises(AuthorizationError) as fault:
            self.run_query(
                "MATCH (p:Patient) WHERE p.age = 42 "
                "SET p.classification = 'normal' FINISH"
            )
        self.assertEqual(fault.exception.code, "42000")
        self.assertEqual(self.graph.to_dict(), before)

        update_policy.with_check = True
        self.graph.nodes["patient_visible"].properties["classification"] = "normal"
        self.catalog.add_policy(
            PolicyDescriptor(
                name="deny_post_update_projection",
                graph="hospital",
                actions={"READ"},
                grantees={"clinician"},
                effect="DENY",
                selector={"kind": "NODE", "labels": ["Patient"]},
                using="RESOURCE.classification = 'restricted'",
            )
        )
        before_projection = self.graph.to_dict()
        with self.assertRaises(AuthorizationError) as projection_denial:
            self.run_query(
                "MATCH (p:Patient) WHERE p.age = 42 "
                "SET p.classification = 'restricted' "
                "RETURN p.classification AS classification"
            )
        self.assertEqual(projection_denial.exception.code, "42000")
        self.assertEqual(self.graph.to_dict(), before_projection)

    def test_remove_uses_old_state_and_checks_new_state(self) -> None:
        update_policy = next(
            policy
            for policy in self.catalog.policies
            if policy.name == "local_patient_update"
        )
        update_policy.with_check = (
            "RESOURCE.address IS NOT NULL AND NEW_RESOURCE.address IS NULL"
        )
        result = self.run_query(
            "MATCH (p:Patient) WHERE p.age = 42 "
            "REMOVE p.address RETURN p.address AS address"
        )
        self.assertEqual(result.rows, [{"address": None}])
        self.assertNotIn("address", self.graph.nodes["patient_visible"].properties)
        self.assertGreater(result.metrics.with_check_evaluations, 0)

        update_policy.with_check = "NEW_RESOURCE.classification <> 'sealed'"
        before = self.graph.to_dict()
        with self.assertRaises(AuthorizationError) as raised:
            self.run_query(
                "MATCH (p:Patient) WHERE p.age = 42 "
                "REMOVE p.classification FINISH"
            )
        self.assertEqual(raised.exception.code, "42000")
        self.assertEqual(self.graph.to_dict(), before)
        self.assertEqual(
            self.graph.nodes["patient_visible"].properties["classification"],
            "normal",
        )

    def test_static_plan_contains_guards_and_property_obligations(self) -> None:
        query = parse_query(
            "MATCH (u:User {login: SESSION_USER})-[:WORKS_AT]->(h:Hospital) "
            "RETURN h.name AS name"
        )
        plan = StaticAnalyzer(self.catalog, "hospital", "alice").logical_plan(query)
        actions = {item["action"] for item in plan["obligations"]}
        self.assertTrue({"ACCESS", "TRAVERSE", "READ"}.issubset(actions))
        self.assertEqual(plan["operators"][1]["operator"], "GuardedMatch")


class PathConfinementTests(unittest.TestCase):
    def test_hidden_intermediate_node_breaks_variable_length_path(self) -> None:
        graph = PropertyGraph.from_dict(
            {
                "name": "g",
                "nodes": [
                    {"id": "s", "labels": ["Start"]},
                    {"id": "x", "labels": ["Secret"]},
                    {"id": "t", "labels": ["Goal"]},
                ],
                "edges": [
                    {"id": "e1", "type": "LINK", "source": "s", "target": "x"},
                    {"id": "e2", "type": "LINK", "source": "x", "target": "t"},
                    {"id": "e3", "type": "BLOCKED", "source": "s", "target": "t"},
                ],
            }
        )
        catalog = AuthorizationCatalog.from_dict(
            {
                "roles": ["reader"],
                "user_roles": {"u": ["reader"]},
                "privileges": [
                    {
                        "grantee": "reader",
                        "actions": ["ACCESS", "MATCH"],
                        "target": {"graph": "g"},
                    }
                ],
                "policies": [
                    {
                        "name": "hide_secret",
                        "graph": "g",
                        "actions": ["TRAVERSE"],
                        "grantees": ["reader"],
                        "effect": "DENY",
                        "selector": {"kind": "NODE", "labels": ["Secret"]},
                        "using": True,
                    },
                    {
                        "name": "hide_blocked_edge",
                        "graph": "g",
                        "actions": ["TRAVERSE"],
                        "grantees": ["reader"],
                        "effect": "DENY",
                        "selector": {"kind": "EDGE", "edge_types": ["BLOCKED"]},
                        "using": True,
                    }
                ],
            }
        )
        result = SecureExecutor(graph, catalog, "u").execute(
            parse_query(
                "MATCH (s:Start)-[:LINK*2..2]->(t:Goal) RETURN ID(t) AS target"
            ),
            compare_reference=True,
        )
        self.assertEqual(result.rows, [])
        self.assertTrue(result.reference_equal)
        self.assertGreater(result.metrics.denied_before_enqueue, 0)
        edge_result = SecureExecutor(graph, catalog, "u").execute(
            parse_query(
                "MATCH (s:Start)-[:BLOCKED]->(t:Goal) RETURN ID(t) AS target"
            ),
            compare_reference=True,
        )
        self.assertEqual(edge_result.rows, [])
        self.assertTrue(edge_result.reference_equal)
        self.assertGreater(edge_result.metrics.denied_before_enqueue, 0)


class UpdateAtomicityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = PropertyGraph("g")
        self.catalog = AuthorizationCatalog.from_dict(
            {
                "roles": ["writer"],
                "user_roles": {"w": ["writer"]},
                "privileges": [
                    {
                        "grantee": "writer",
                        "actions": [
                            "ACCESS",
                            "MATCH",
                            "INSERT",
                            "DELETE",
                            "SET",
                            "REMOVE",
                        ],
                        "target": {"graph": "g"},
                    }
                ],
                "policies": [
                    {
                        "name": "deny_bad_insert",
                        "graph": "g",
                        "actions": ["INSERT"],
                        "grantees": ["writer"],
                        "effect": "DENY",
                        "selector": {"kind": "NODE", "labels": ["Denied"]},
                        "with_check": True,
                    }
                ],
            }
        )

    def execute(self, source: str):
        return SecureExecutor(self.graph, self.catalog, "w").execute(parse_query(source))

    def test_multi_element_insert_is_all_or_nothing(self) -> None:
        deny_policy = next(
            policy
            for policy in self.catalog.policies
            if policy.name == "deny_bad_insert"
        )
        deny_policy.with_check = "RESOURCE IS NULL AND NEW_RESOURCE:Denied"
        before = self.graph.to_dict()
        with self.assertRaises(AuthorizationError) as raised:
            self.execute(
                "INSERT (a:Allowed {_id:'a'}), (b:Denied {_id:'b'}) FINISH"
            )
        self.assertEqual(raised.exception.code, "42000")
        self.assertEqual(self.graph.to_dict(), before)

        deny_policy.with_check = "RESOURCE.classification = 'blocked'"
        with self.assertRaises(AuthorizationError) as unknown_denial:
            self.execute(
                "INSERT (b:Denied {_id:'b', classification:'clear'}) FINISH"
            )
        self.assertEqual(unknown_denial.exception.code, "42000")
        self.assertEqual(self.graph.to_dict(), before)

    def test_inserted_edge_and_endpoints_commit_together(self) -> None:
        result = self.execute(
            "INSERT (a:A {_id:'a'})-[e:LINK {_id:'e'}]->(b:B {_id:'b'}) "
            "RETURN ID(e) AS edge"
        )
        self.assertEqual(result.rows, [{"edge": "e"}])
        self.assertEqual(result.affected_elements, 3)
        self.assertEqual(self.graph.edges["e"].source, "a")
        self.assertEqual(self.graph.edges["e"].target, "b")

    def test_plain_delete_is_nodetach_and_detach_is_explicit(self) -> None:
        self.execute("INSERT (a:A {_id:'a'})-[:LINK {_id:'e'}]->(b:B {_id:'b'}) FINISH")
        before = self.graph.to_dict()
        with self.assertRaises(ExecutionError) as raised:
            self.execute("MATCH (a:A) DELETE a FINISH")
        self.assertEqual(raised.exception.code, "G1001")
        self.assertEqual(self.graph.to_dict(), before)
        result = self.execute("MATCH (a:A) DETACH DELETE a FINISH")
        self.assertEqual(result.affected_elements, 2)
        self.assertNotIn("a", self.graph.nodes)
        self.assertNotIn("e", self.graph.edges)

    def test_delete_policy_failure_precedes_nodetach_validation(self) -> None:
        graph = PropertyGraph.from_dict(
            {
                "name": "g",
                "nodes": [
                    {"id": "a", "labels": ["Protected"]},
                    {"id": "b", "labels": ["Other"]},
                ],
                "edges": [
                    {"id": "e", "type": "LINK", "source": "a", "target": "b"}
                ],
            }
        )
        catalog = AuthorizationCatalog.from_dict(
            {
                "roles": ["writer"],
                "user_roles": {"w": ["writer"]},
                "privileges": [
                    {
                        "grantee": "writer",
                        "actions": ["ACCESS", "MATCH", "DELETE"],
                        "target": {"graph": "g"},
                    }
                ],
                "policies": [
                    {
                        "name": "protect_delete",
                        "graph": "g",
                        "actions": ["DELETE"],
                        "grantees": ["writer"],
                        "effect": "DENY",
                        "selector": {"kind": "NODE", "labels": ["Protected"]},
                        "using": True,
                    }
                ],
            }
        )
        with self.assertRaises(AuthorizationError) as raised:
            SecureExecutor(graph, catalog, "w").execute(
                parse_query("MATCH (a:Protected) DELETE a FINISH")
            )
        self.assertEqual(raised.exception.code, "42000")
        self.assertIn("a", graph.nodes)
        self.assertIn("e", graph.edges)
        before = graph.to_dict()
        with self.assertRaises(AuthorizationError) as detach_raised:
            SecureExecutor(graph, catalog, "w").execute(
                parse_query("MATCH (a:Protected) DETACH DELETE a FINISH")
            )
        self.assertEqual(detach_raised.exception.code, "42000")
        self.assertEqual(graph.to_dict(), before)


class PolicyLanguageTests(unittest.TestCase):
    def test_article_style_exists_predicate_uses_session_user(self) -> None:
        graph = PropertyGraph.load(EXAMPLES / "hospital.graph.json")
        predicate = compile_predicate(
            "EXISTS { "
            "MATCH (u:User {login: SESSION_USER})-[:WORKS_AT]->(h:Hospital) "
            "MATCH (RESOURCE:Patient)-[:TREATED_AT]->(h) "
            "}"
        )
        context = PolicyEvaluationContext(
            "alice", {"clinician"}, resource=graph.nodes["patient_visible"]
        )
        self.assertTrue(predicate(graph, context))
        context.resource = graph.nodes["patient_remote"]
        self.assertFalse(predicate(graph, context))

        context.resource = graph.nodes["patient_visible"]
        self.assertIsNone(compile_predicate("NULL = NULL")(graph, context))
        self.assertIsNone(compile_predicate("NULL <> NULL")(graph, context))
        self.assertFalse(compile_predicate("NULL IN ()")(graph, context))
        self.assertIsNone(
            compile_predicate("NOT (RESOURCE.missing = 1)")(graph, context)
        )
        self.assertFalse(
            compile_predicate("(RESOURCE.missing = 1) AND FALSE")(graph, context)
        )
        self.assertIsNone(
            compile_predicate("(RESOURCE.missing = 1) AND TRUE")(graph, context)
        )
        self.assertTrue(
            compile_predicate("(RESOURCE.missing = 1) OR TRUE")(graph, context)
        )
        self.assertIsNone(
            compile_predicate("(RESOURCE.missing = 1) OR FALSE")(graph, context)
        )
        self.assertTrue(
            compile_predicate(
                "RESOURCE.classification IN ('normal', NULL)"
            )(graph, context)
        )
        self.assertIsNone(
            compile_predicate(
                "RESOURCE.classification IN ('sealed', NULL)"
            )(graph, context)
        )
        self.assertFalse(
            compile_predicate(
                "EXISTS { MATCH (p:Patient) WHERE p.missing = 1 }"
            )(graph, context)
        )
        with self.assertRaises(CatalogError):
            compile_predicate("EXISTS { INSERT (n:N) FINISH }")
        with self.assertRaises(CatalogError):
            compile_predicate(
                "EXISTS { MATCH (n:Patient) RETURN n LIMIT 0 }"
            )
        with self.assertRaises(CatalogError):
            compile_predicate("EXISTS { MATCH (n:Patient) FINISH }")

        unknown_json = {
            "property_compare": {
                "source": "RESOURCE",
                "property": "missing",
                "op": "=",
                "value": "present",
            }
        }
        self.assertIsNone(compile_predicate(unknown_json)(graph, context))
        self.assertFalse(
            compile_predicate({"all": [unknown_json, False]})(graph, context)
        )
        self.assertIsNone(
            compile_predicate({"all": [unknown_json, True]})(graph, context)
        )
        self.assertTrue(
            compile_predicate({"any": [unknown_json, True]})(graph, context)
        )
        self.assertIsNone(
            compile_predicate({"any": [unknown_json, False]})(graph, context)
        )
        self.assertIsNone(
            compile_predicate({"not": unknown_json})(graph, context)
        )


if __name__ == "__main__":
    unittest.main()
