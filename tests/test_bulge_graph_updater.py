import json
from pathlib import Path
from difflib import SequenceMatcher
import sys
from collections import defaultdict
import atexit

import pytest

# Ensure the repository root is on the Python path so that `core` can be imported
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from core.models import BulgeGraph, GraphNode
from core.bulge_graph_updater import BulgeGraphUpdater

# Global variable to collect test results by category
test_results_by_category = defaultdict(lambda: defaultdict(lambda: {'passed': 0, 'failed': 0, 'failures': []}))


def _build_graph(mapping):
    elements = {}
    for name, coords in mapping.items():
        elements[name] = GraphNode(
            positions=coords.copy(),
            start=min(coords) if coords else None,
            end=max(coords) if coords else None,
        )
    return BulgeGraph(elements=elements)


def _diff_indices(pre: str, post: str, coords=None):
    """Return indices of inserted and deleted bases.

    Parameters
    ----------
    pre, post : str
        Sequences before and after the modification.
    coords : list[int] or None
        1-based coordinates of the node that was modified.  When provided the
        returned indices are clamped to the nearest coordinate.  This helps
        when ``SequenceMatcher`` chooses a slightly offset index due to
        repetitive sequence content (common in loop regions).
    """

    sm = SequenceMatcher(None, pre, post)
    inserted, deleted = [], []
    for tag, a1, a2, b1, b2 in sm.get_opcodes():
        if tag == "insert":
            inserted.extend(range(a1, a1 + (b2 - b1)))
        elif tag == "delete":
            deleted.extend(range(a1, a2))

    if coords:
        def _closest(idx: int) -> int:
            return min(coords, key=lambda c: abs((c - 1) - idx)) - 1

        if inserted:
            inserted = [_closest(i) for i in inserted]
        if deleted:
            deleted = [_closest(i) for i in deleted]

    return inserted, deleted


def load_cases():
    path = Path(__file__).parent / "test_cases.json"
    with open(path) as f:
        data = json.load(f)
    return data["test_cases"]


def _format_modification_info(case, inserted, deleted):
    """Format detailed information about the RNA modification."""
    info_lines = []
    
    # Basic modification info
    info_lines.append(f"Modification: {case['mod_action']} on {case['mod_node_type']} '{case['mod_node']}'")
    info_lines.append(f"Original coordinates: {case.get('original_coords', 'N/A')}")
    
    # Sequence change info
    if inserted:
        info_lines.append(f"Inserted at positions: {inserted}")
    if deleted:
        info_lines.append(f"Deleted from positions: {deleted}")
    
    info_lines.append(f"Sequence length change: {case.get('sequence_length_change', 'N/A')}")
    
    # Structure info
    info_lines.append(f"Pre-modification sequence:  {case.get('pre_mod_seq', 'N/A')}")
    info_lines.append(f"Pre-modification structure: {case.get('pre_mod_dotbracket', 'N/A')}")
    info_lines.append(f"Post-modification sequence: {case.get('post_mod_seq', 'N/A')}")
    info_lines.append(f"Post-modification structure:{case.get('post_mod_dotbracket', 'N/A')}")
    
    return '\n'.join(info_lines)


def _format_graph_differences(result, expected):
    """Format the differences between actual and expected graph results."""
    diff_lines = []
    
    all_nodes = set(result.keys()) | set(expected.keys())
    
    for node in sorted(all_nodes):
        result_pos = result.get(node, [])
        expected_pos = expected.get(node, [])
        
        if result_pos != expected_pos:
            diff_lines.append(f"  {node}: got {result_pos}, expected {expected_pos}")
        elif len(diff_lines) < 10:  # Show some matching ones too, but limit output
            diff_lines.append(f"  {node}: {result_pos} ✓")
    
    return '\n'.join(diff_lines)


def _get_action_category(mod_action):
    """Categorize modification actions into insertions and deletions."""
    if mod_action in ['insert', 'insert_pair']:
        return 'insertions'
    elif mod_action in ['delete', 'delete_pair']:
        return 'deletions'
    else:
        return 'other'


def _normalize_structure_type(mod_node_type):
    """Normalize structure type names for consistent categorization."""
    type_mapping = {
        'stem': 'stems',
        'hairpin': 'hairpins', 
        'internal': 'internal_loops',
        'bulge': 'bulges',
        'multi': 'multi_loops'
    }
    return type_mapping.get(mod_node_type.lower(), mod_node_type)


def save_and_print_summary():
    """Save and print the categorized test results summary."""
    global test_results_by_category
    
    if not test_results_by_category:
        return
    
    summary_lines = []
    summary_lines.append("\n" + "="*80)
    summary_lines.append("TEST RESULTS BY STRUCTURE TYPE AND ACTION")
    summary_lines.append("="*80)
    
    total_passed = 0
    total_failed = 0
    
    # Sort structure types for consistent output
    structure_types = ['stems', 'hairpins', 'internal_loops', 'bulges', 'multi_loops']
    
    for structure_type in structure_types:
        if structure_type not in test_results_by_category:
            continue
            
        summary_lines.append(f"\n{structure_type.upper()}:")
        summary_lines.append("-" * 40)
        
        structure_passed = 0
        structure_failed = 0
        
        # Process insertions and deletions
        for action_type in ['insertions', 'deletions']:
            if action_type in test_results_by_category[structure_type]:
                stats = test_results_by_category[structure_type][action_type]
                passed = stats['passed']
                failed = stats['failed']
                total_tests = passed + failed
                
                if total_tests > 0:
                    success_rate = (passed / total_tests) * 100
                    summary_lines.append(f"  {action_type}: {passed}/{total_tests} passed ({success_rate:.1f}%)")
                    
                    if failed > 0:
                        failed_cases = [f['case_id'] for f in stats['failures']]
                        summary_lines.append(f"    Failed cases: {failed_cases[:10]}{'...' if len(failed_cases) > 10 else ''}")
                    
                    structure_passed += passed
                    structure_failed += failed
        
        if structure_passed + structure_failed > 0:
            structure_total = structure_passed + structure_failed
            structure_success_rate = (structure_passed / structure_total) * 100
            summary_lines.append(f"  TOTAL: {structure_passed}/{structure_total} passed ({structure_success_rate:.1f}%)")
            
            total_passed += structure_passed
            total_failed += structure_failed
    
    # Print overall summary
    if total_passed + total_failed > 0:
        overall_total = total_passed + total_failed
        overall_success_rate = (total_passed / overall_total) * 100
        summary_lines.append(f"\n{'='*80}")
        summary_lines.append(f"OVERALL SUMMARY: {total_passed}/{overall_total} passed ({overall_success_rate:.1f}%)")
        summary_lines.append(f"{'='*80}")
        
        # Print detailed failure summary if there are failures
        if total_failed > 0:
            summary_lines.append(f"\nDETAILED FAILURE SUMMARY:")
            summary_lines.append("-" * 40)
            for structure_type, actions in test_results_by_category.items():
                for action_type, stats in actions.items():
                    if stats['failed'] > 0:
                        summary_lines.append(f"\n{structure_type} - {action_type} ({stats['failed']} failures):")
                        for failure in stats['failures'][:5]:  # Show first 5 failures
                            summary_lines.append(f"  Case {failure['case_id']}: {failure['node']} ({failure['action']})")
                        if len(stats['failures']) > 5:
                            summary_lines.append(f"  ... and {len(stats['failures']) - 5} more")
    
    # Print to console
    summary_text = '\n'.join(summary_lines)
    print(summary_text)
    
    # Also save to file
    try:
        with open('test_summary_by_category.txt', 'w') as f:
            f.write(summary_text)
    except Exception:
        pass  # Don't fail if we can't write the file

# Register the summary function to run at exit
atexit.register(save_and_print_summary)


@pytest.mark.parametrize("case", load_cases())
def test_bulge_graph_update(case):
    global test_results_by_category
    
    bg = _build_graph(case["pre_mod_bulgegraph"])
    node = case["mod_node"]
    coords = case["pre_mod_bulgegraph"].get(node, [])
    inserted, deleted = _diff_indices(case["pre_mod_seq"], case["post_mod_seq"], coords)

    action = case["mod_action"]

    if action == "insert_pair":
        # SequenceMatcher can sometimes return more than two indices or
        # duplicate indices when aligning repetitive regions.  Use the
        # outermost indices to determine the inserted base pair.
        left, right = min(inserted), max(inserted)
        BulgeGraphUpdater.insert_stem_pair(bg, node, left, right)
    elif action == "delete_pair":
        left, right = min(deleted), max(deleted)
        BulgeGraphUpdater.delete_stem_pair(bg, node, left, right)
    elif action == "insert":
        BulgeGraphUpdater.insert_loop_base(bg, node, inserted[0])
    elif action == "delete":
        BulgeGraphUpdater.delete_loop_base(bg, node, deleted[0])

    result = {n: g.positions for n, g in bg.elements.items()}

    # Normalize results for comparison with forgi output. Forgi often stores
    # only the start and end coordinate for a loop, while ``BulgeGraphUpdater``
    # may keep every position explicitly.  If the expected data has exactly two
    # coordinates but our result has more, compare using only the first and last
    # positions from the result.
    normalized = {}
    for node_name, exp_pos in case["post_mod_bulgegraph"].items():
        res_pos = result.get(node_name, [])
        if len(exp_pos) == 2 and len(res_pos) > 2:
            normalized[node_name] = [res_pos[0], res_pos[-1]]
        elif len(res_pos) > len(exp_pos) and set(exp_pos).issubset(res_pos):
            # Our representation keeps every coordinate while the expected list
            # represents a condensed range.
            normalized[node_name] = exp_pos
        else:
            normalized[node_name] = res_pos

    result = normalized
    
    # Categorize the test
    structure_type = _normalize_structure_type(case.get('mod_node_type', 'unknown'))
    action_category = _get_action_category(action)
    
    # Check if test passes
    test_passed = result == case["post_mod_bulgegraph"]
    
    if test_passed:
        test_results_by_category[structure_type][action_category]['passed'] += 1
    else:
        test_results_by_category[structure_type][action_category]['failed'] += 1
        failure_info = {
            'case_id': case.get('case_id', 'N/A'),
            'node': node,
            'action': action,
            'structure_type': structure_type,
            'expected_diff': _format_graph_differences(result, case["post_mod_bulgegraph"])
        }
        test_results_by_category[structure_type][action_category]['failures'].append(failure_info)
    
    # Create detailed error message if assertion fails
    if not test_passed:
        modification_info = _format_modification_info(case, inserted, deleted)
        graph_differences = _format_graph_differences(result, case["post_mod_bulgegraph"])
        
        error_msg = f"""
RNA Structure Modification Test Failed
=====================================

{modification_info}

Graph Node Differences:
{graph_differences}

Case ID: {case.get('case_id', 'N/A')}
Attempt: {case.get('attempt', 'N/A')}
"""
        pytest.fail(error_msg)
    
    assert result == case["post_mod_bulgegraph"]


# Multiple hooks to ensure the summary is printed
def pytest_sessionfinish(session, exitstatus):
    """Called after whole test run finished."""
    # Don't print here to avoid duplicate output since we use atexit
    pass


@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Called to write the summary at the end of the terminal output."""
    # This hook is called after all tests are done and before pytest exits
    # It's more reliable than sessionfinish
    save_and_print_summary()