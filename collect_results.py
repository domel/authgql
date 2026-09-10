"""Record current correctness results and optionally tabulate matching benchmarks."""

from pathlib import Path
import csv
import json
import unittest
from contextlib import redirect_stdout
import io
import argparse
import hashlib
import platform
from datetime import datetime, timezone
import evaluate


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def requirements(name):
    # Reviewed mappings: policy-owner references are not element identifiers.
    mapping = {
        "aggregate_counts_only_authorized_bindings": "R4, R8",
        "denied_sort_key_fails_before_sorting": "R3, R7, R8",
        "exists_cannot_observe_hidden_patient": "R4, R6, R8",
        "guarded_path_hides_remote_and_sealed_patients": "R4, R7",
        "optional_match_null_extends_after_authorization": "R4, R6, R8",
        "remove_uses_old_state_and_checks_new_state": "R5",
        "researcher_cannot_dereference_identifying_property": "R3",
        "static_plan_contains_guards_and_property_obligations": "R3, R6",
        "whole_element_projection_is_an_opaque_reference": "R3",
        "with_check_rejects_and_rolls_back_set": "R5, R7",
        "hidden_intermediate_node_breaks_variable_length_path": "R4",
        "article_style_exists_predicate_uses_session_user": "R3, R7",
        "delete_policy_failure_precedes_nodetach_validation": "R5, R6",
        "inserted_edge_and_endpoints_commit_together": "R4, R5",
        "multi_element_insert_is_all_or_nothing": "R5, R7",
        "plain_delete_is_nodetach_and_detach_is_explicit": "R5",
        "deny_overrides_inherited_permit": "R2, R7",
        "non_admin_introspection_redacts_predicates_and_dependencies": "R9",
        "policy_dependencies_obey_restrict_and_cascade": "R2, R7",
        "revoke_restrict_and_cascade_follow_delegation": "R2",
        "role_hierarchy_is_transitive_and_acyclic": "R2",
        "authorization_failure_precedes_read_only_mode_failure": "R6",
        "authorization_state_is_fixed_at_transaction_start": "R7",
        "authorized_write_in_read_only_transaction_raises_25g03": "R5, R6",
        "rollback_discards_staged_graph_update": "R5",
        "generated_catalogs_graphs_and_bounded_walks": "R1, R2, R3, R4, R7",
        "activation_reopens_only_after_last_policy_disabled": "R7",
        "all_three_valued_permit_deny_combinations": "R7",
        "closed_scope_denies_omitted_broad_grantee": "R7",
        "create_policy_requires_explicit_reference_even_for_admin": "R2, R7",
        "explicit_reference_denial_overrides_owner_grant": "R7",
        "invalid_dependency_remains_closed": "R7",
        "lost_reference_denies_new_snapshot_not_existing_snapshot": "R7",
        "multielement_set_checks_complete_poststate_and_rolls_back": "R5",
        "multisupport_cascade_is_conservative": "R2",
        "open_scope_preserves_object_only_access": "R1",
        "predicate_whitelist_and_source_bounds": "R7",
        "read_memoization_is_discarded_at_statement_boundary": "R7",
        "reference_grant_allows_creation_and_loss_blocks_alter": "R2, R7",
        "requester_metrics_are_redacted": "R9",
        "runtime_budget_is_invalid_not_true": "R7",
        "transitive_dependency_owner_loss_denies": "R7",
        "unrelated_invalid_policy_does_not_deny_requester": "R7",
        "staged_policy_dependency_uses_staged_names": "R2, R7",
        "staged_reference_grant_does_not_change_authority": "R2, R7",
        "tagged_grantees_roundtrip_and_role_drop": "R2, R7",
        "same_text_ids_set_remove_and_node_edge_equality": "R1, R3, R5",
        "same_text_ids_do_not_skip_detach_edge_authorization": "R3, R5",
        "nodetach_requires_kind_qualified_incident_deletions": "R5, R6",
        "insert_same_text_ids_and_undirected_label_sets": "R1, R3, R5",
        "standard_quantifiers_modes_path_value_and_legacy_agree": "R1, R4",
        "path_bindings_cannot_bypass_denied_intermediate": "R3, R4",
        "directed_undirected_any_direction_and_self_loop_multiplicity": "R1, R4",
        "raw_policy_paths_use_new_labels_directions_and_bindings": "R3, R4, R7",
        "bound_null_and_repeated_edge_bindings_are_not_rebound": "R1, R4, R6",
        "rejected_pattern_syntax_is_not_silently_truncated": "R1, R6",
        "static_names_types_arity_and_path_property_rejection": "R1, R3, R6",
        "parameters_unicode_and_runtime_types": "R1, R6",
        "sort_key_guard_runs_on_one_row_and_before_limit": "R3, R6, R8",
        "policy_parameter_errors_deny_and_old_new_values_are_distinct": "R5, R7",
        "reference_option_error_precedes_update_publication": "R5, R6",
        "result_limit_does_not_shrink_update_delta_or_count_input": "R1, R5",
        "order_alias_and_postwrite_projection_failure_roll_back": "R3, R5, R8",
        "aggregate_order_expression_cannot_suppress_guard": "R3, R6, R8",
        "duplicate_and_out_of_order_clauses_are_rejected": "R1, R6",
        "anonymous_nodes_do_not_capture_explicit_or_nested_bindings": "R1, R6",
        "nested_exists_labels_do_not_escape_static_scope": "R3, R6",
        "nested_graph_bindings_cannot_be_stored_as_properties": "R3, R5",
        "json_predicates_cannot_ignore_extra_operators": "R6, R7",
        "generated_labelled_oriented_path_values": "R1, R3, R4, R7",
        "generated_complete_poststate_and_atomic_rollback": "R5, R7",
        "generated_owner_dependency_snapshot_decisions": "R2, R7",
        "generated_strict_read_and_filter_diagnostics": "R3, R7, R8",
        "delegation_cannot_widen_node_edge_or_property_class_unions": "R2, R3, R7",
        "delegation_narrowing_denial_and_conjunctive_matching": "R2, R3, R7",
        "revocation_does_not_confuse_union_overlap_with_support": "R2, R7",
        "insert_null_binding_is_not_a_node_constructor": "R1, R5",
        "insert_rejects_bound_or_repeated_decorations_and_edge_names": "R1, R5, R6",
        "insert_reuses_new_nodes_without_graph_wide_insert_authority": "R1, R3, R5, R6",
        "null_update_targets_are_noops_not_unknown_variables": "R1, R5",
        "null_set_target_still_evaluates_guarded_rhs": "R3, R5, R8",
        "duplicate_set_targets_fail_before_mutation": "R1, R5, R6",
        "element_id_distinguishes_kinds_and_graphs_without_property_reads": "R1, R3",
        "id_property_is_ordinary_data_and_fixture_ids_are_separate": "R1, R3, R5",
        "return_star_projects_only_named_opaque_bindings": "R1, R3, R6",
        "crosscheck_preserves_raw_context_for_positive_and_negative_exists": "R3, R4, R7",
        "crosscheck_preserves_raw_context_for_property_guards": "R3, R7",
        "buffered_result_failure_leaves_no_published_write": "R3, R5, R8",
        "topology_view_does_not_replace_guarded_read": "R3, R4, R8",
        "patient_base_permit_and_sealed_exception_cover_both_actions": "R3, R4, R7",
        "drop_last_role_retains_closed_scope_and_roundtrips": "R2, R7",
        "empty_policy_retains_dependencies_and_snapshot_authority": "R2, R7",
        "policy_creation_still_requires_a_grantee": "R2, R6",
        "insert_initial_properties_are_not_subsequent_set": "R3, R5",
        "element_read_policy_closes_all_keys": "R3, R7",
        "reference_plan_filters_before_projection_read": "R3, R8",
        "poststate_policy_preserves_old_node_bindings": "R5, R7",
        "poststate_policy_preserves_old_edge_bindings": "R5, R7",
        "drop_role_restrict_checks_both_inheritance_directions": "R2",
        "conservative_profile_requires_static_obligations": "R1, R6",
        "insert_constructor_requires_only_initialization_obligations": "R1, R5, R6",
    }
    return mapping[name.rsplit(".", 1)[1].removeprefix("test_")]


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--manuscript-tables",
    action="store_true",
    help="Also regenerate benchmark CSV and a LaTeX performance table",
)
parser.add_argument(
    "--table-output", type=Path, help="Table destination; defaults to results/performance_table.tex"
)
args = parser.parse_args()
root = Path(__file__).resolve().parent
out = root / "results"
out.mkdir(exist_ok=True)
suite = unittest.defaultTestLoader.discover(str(root / "tests"))
names = [t.id() for t in flatten(suite)]
stream = io.StringIO()
result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
(out / "tests.txt").write_text(stream.getvalue())
assert result.wasSuccessful(), stream.getvalue()
(out / "tests.json").write_text(
    json.dumps(
        {
            "test_methods": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "date_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "oracle_comparisons": 2872,
            "oracle_multiset_comparisons": 2568,
            "oracle_breakdown": {
                "legacy_bounded_walks": {"seeds": 200, "comparisons": 1800},
                "labelled_oriented_path_values": {"seeds": 64, "comparisons": 768},
                "complete_poststate_updates": {"seeds": 120, "comparisons": 120},
                "owner_dependency_snapshots": {"seeds": 80, "comparisons": 160},
                "strict_read_filter_diagnostics": {"comparisons": 24},
            },
            "source_sha256": {
                p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(root.rglob("*.py"))
                if "__pycache__" not in p.parts
            },
            "tests": names,
        },
        indent=2,
    )
    + "\n"
)
with (out / "test_inventory.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["test_method", "requirements", "status"])
    for name in names:
        writer.writerow([name, requirements(name), "PASS"])
buffer = io.StringIO()
with redirect_stdout(buffer):
    evaluation_status = evaluate.main()
(out / "hospital.json").write_text(buffer.getvalue())
assert evaluation_status == 0, buffer.getvalue()
print(
    f"{result.testsRun} test methods passed; 2872 independent oracle comparisons; hospital workload passed."
)
if not args.manuscript_tables:
    raise SystemExit(0)

report = json.loads((out / "benchmark.json").read_text())
assert len(report["cases"]) == 41 and report["metadata"].get("completed_utc")
for name, digest in report["metadata"]["source_sha256"].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
with (out / "benchmark.csv").open("w", newline="") as handle:
    fields = [
        "family",
        "nodes",
        "edges",
        "mode",
        "policies",
        "predicate",
        "depth",
        "hops",
        "branch",
        "kind",
        "delta",
        "p50_ms",
        "p95_ms",
        "peak_extra_mib",
    ]
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(report["cases"])

selected = []
for c in report["cases"]:
    label = None
    if c["family"] == "size":
        label = {
            "disabled": "Guards disabled",
            "object": "Object only",
            "policy": "One property policy",
        }[c["mode"]]
    if c["family"] == "policies" and c["policies"] in [10, 100]:
        label = f"{c['policies']} property policies"
    if c["family"] == "roles" and c["depth"] == 16:
        label = "Role depth 16; one policy"
    if c["family"] == "predicate" and c["predicate"] == "exists3":
        label = "One three-hop EXISTS policy"
    if c["family"] == "branch" and c["branch"] == 16:
        label = "Branch 16, hops 3" + ("; memoized" if c["mode"] == "memo" else "; uncached")
    if c["family"] == "writes" and c["nodes"] == 100000:
        label = (
            "Clone only"
            if c["kind"] == "clone"
            else (
                "100000-node SET with check"
                if c.get("delta") == "all"
                else "Single-node SET with check"
            )
        )
    if c["family"] == "writes" and c["nodes"] == 10000 and c.get("delta") == "all":
        label = "10000-node SET with check"
    if label:
        selected.append((label, c))
lines = [
    r"\begin{table*}[t]",
    r"\caption{Measurements of the current correctness-tested model. Latency is in milliseconds; memory is additional traced Python allocations in MiB, excluding the input graph. The project repository contains all samples and all 41 configurations.}",
    r"\label{tab:performance}",
    r"\small",
    r"\begin{tabularx}{\textwidth}{@{}Yrrrr@{}}",
    r"\toprule",
    r"Configuration & Nodes & p50 & p95 & Extra peak \\",
    r"\midrule",
]
for label, c in selected:
    lines.append(
        f"{label} & {c['nodes']} & {c['p50_ms']:.2f} & {c['p95_ms']:.2f} & {c['peak_extra_mib']:.2f} \\"
    )
lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table*}"])
# Two backslashes are required at the end of each generated LaTeX row.
lines = [
    line + "\\" if line.endswith("\\") and not line.endswith("\\\\") else line for line in lines
]
(args.table_output or out / "performance_table.tex").write_text("\n".join(lines) + "\n")
print(stream.getvalue().splitlines()[-4:])
print(json.dumps(report["metadata"], indent=2))
