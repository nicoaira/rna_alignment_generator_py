#!/usr/bin/env python3
"""Interactive RNA alignment mutation viewer.

This auxiliary script mirrors the alignment generator pipeline but exposes
an interactive step-by-step view of each modification cycle so users can
inspect sampled parameters, structural edits, and resulting sequences in
real time.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from config.cli_parser import parse_arguments as parse_generator_arguments

# Provide a minimal tqdm stub if the optional dependency is missing. The core
# alignment generator imports tqdm for progress bars, but the interactive tool
# does not rely on it.
try:  # pragma: no cover - dependency shim
    import tqdm  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - fallback path
    import types

    dummy_module = types.ModuleType("tqdm")

    class _DummyTqdm:
        def __init__(self, *args, **kwargs):
            iterable = kwargs.get('iterable') if 'iterable' in kwargs else (args[0] if args else None)
            self._iterable = iterable if iterable is not None else []

        def __iter__(self):
            return iter(self._iterable)

        def update(self, *args, **kwargs):
            return None

        def close(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def _dummy_tqdm(*args, **kwargs):
        return _DummyTqdm(*args, **kwargs)

    dummy_module.tqdm = _dummy_tqdm
    sys.modules['tqdm'] = dummy_module

from core.alignment_generator import (
    AlignmentMutationEngine,
    _complement,
    _pair_map_from_structure,
    _random_base_except,
)
from core.models import (
    ActionCounts,
    BulgeGraph,
    GraphNode,
    ModificationCounts,
    NodeType,
    SampledModifications,
    classify_node,
)
from core.rna_generator import BulgeGraphParser, RnaGenerator

logger = logging.getLogger(__name__)


@dataclass
class InteractiveNode:
    """State of a sequence/structure along the interactive path."""

    path: str
    sequence: str
    structure: str
    bulge_graph: BulgeGraph
    col_map: List[int]


@dataclass
class EventInfo:
    """Human-readable description of a single mutation event."""

    event_type: str
    description: str
    detail: Dict[str, Any]


@dataclass
class BranchResult:
    """Result of mutating the current node into one child branch."""

    node: InteractiveNode
    sampled_mods: SampledModifications
    mod_counts: ModificationCounts
    action_counts: ActionCounts
    events: List[EventInfo]
    substitution_positions: List[int]
    substitution_columns: List[int]
    parent_sequence: str
    parent_structure: str
    parent_col_map: List[int]


def wrap_text(text: str, width: int = 80) -> str:
    """Wrap long strings for terminal readability."""
    if not text:
        return text
    return "\n".join(textwrap.wrap(text, width))


def deepcopy_bulge_graph(graph: BulgeGraph) -> BulgeGraph:
    """Create a deepcopy of a BulgeGraph without relying on copy.deepcopy."""
    return BulgeGraph(
        elements={
            name: GraphNode(positions=list(node.positions), start=node.start, end=node.end)
            for name, node in graph.elements.items()
        }
    )


class RealTimeAlignmentRunner:
    """Driver that mirrors the dataset generator while exposing internal events."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.rna_generator = RnaGenerator()
        self.bulge_parser = BulgeGraphParser()
        self.next_column_id: int = 0

    # ------------------------------------------------------------------
    # Root initialisation & conserved-site sampling
    # ------------------------------------------------------------------
    def create_root(self) -> InteractiveNode:
        """Generate the ancestral sequence/structure used as the starting point."""
        length = self.rna_generator.choose_sequence_length(
            self.args.seq_len_distribution,
            self.args.seq_min_len,
            self.args.seq_max_len,
            self.args.seq_len_mean,
            self.args.seq_len_sd,
        )
        sequence = self.rna_generator.generate_random_sequence(length)
        structure = self.rna_generator.fold_rna(sequence)
        bulge = self.bulge_parser.parse_structure(structure)
        col_map = list(range(length))
        self.next_column_id = len(col_map)
        return InteractiveNode(path="", sequence=sequence, structure=structure, bulge_graph=bulge, col_map=col_map)

    def select_conserved_sites(self, structure: str) -> Tuple[set[int], List[Tuple[int, int]]]:
        """Select conserved single columns and paired columns (copy of dataset logic)."""
        n = len(structure)
        if n == 0:
            return set(), []

        target_sites = max(0, int(round(self.args.f_conserved_sites * n)))
        if target_sites == 0:
            return set(), []

        pair_map = _pair_map_from_structure(structure)
        uniq_pairs: List[Tuple[int, int]] = []
        seen = set()
        for i, j in pair_map.items():
            if i < j and (i, j) not in seen and (j, i) not in seen:
                seen.add((i, j))
                uniq_pairs.append((i, j))
        uniq_pairs = [(i, j) for (i, j) in uniq_pairs if (j - i - 1) >= 3]
        random.shuffle(uniq_pairs)

        single_indices: List[int] = [idx for idx, ch in enumerate(structure) if ch == '.']
        random.shuffle(single_indices)

        chosen_cols: set[int] = set()
        chosen_pairs: List[Tuple[int, int]] = []
        pair_idx = 0
        single_idx = 0

        while len(chosen_cols) < target_sites:
            remaining_pairs = len(uniq_pairs) - pair_idx
            remaining_singles = len(single_indices) - single_idx
            if remaining_pairs <= 0 and remaining_singles <= 0:
                break

            remaining_needed = target_sites - len(chosen_cols)
            if remaining_needed == 1 and remaining_singles > 0:
                idx = single_indices[single_idx]
                single_idx += 1
                chosen_cols.add(idx)
                continue

            options: List[Tuple[str, int]] = []
            if remaining_singles > 0:
                options.append(("single", remaining_singles))
            if remaining_pairs > 0:
                options.append(("pair", remaining_pairs * 2))
            if not options:
                break

            labels = [label for label, _ in options]
            weights = [weight for _, weight in options]
            choice = random.choices(labels, weights=weights, k=1)[0]

            if choice == "single":
                idx = single_indices[single_idx]
                single_idx += 1
                chosen_cols.add(idx)
            else:
                i, j = uniq_pairs[pair_idx]
                pair_idx += 1
                chosen_pairs.append((i, j))
                chosen_cols.add(i)
                chosen_cols.add(j)

        return chosen_cols, chosen_pairs

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------
    def _apply_substitutions(
        self,
        sequence: str,
        structure: str,
        conserved_cols: set[int],
        pair_override: Optional[Dict[int, int]] = None,
    ) -> Tuple[str, List[int]]:
        """Apply substitutions while respecting conserved stems.

        Returns the mutated sequence and a list of 0-based indices that changed.
        """
        if not sequence:
            return sequence, []
        n = len(sequence)
        n_subs = max(0, int(round(n * self.args.f_substitution_rate)))
        if n_subs == 0:
            return sequence, []
        pair_map = pair_override if pair_override is not None else _pair_map_from_structure(structure)
        indices = list(range(n))
        random.shuffle(indices)
        taken = set()
        seq_list = list(sequence)
        changed: List[int] = []
        for idx in indices:
            if len(taken) >= n_subs:
                break
            if idx in taken:
                continue
            base = seq_list[idx]
            if idx in conserved_cols and idx in pair_map:
                partner = pair_map[idx]
                if partner in taken:
                    continue
                new_base = _random_base_except(base)
                if new_base != base:
                    seq_list[idx] = new_base
                    changed.append(idx)
                comp = _complement(new_base)
                if partner < len(seq_list) and seq_list[partner] != comp:
                    seq_list[partner] = comp
                    changed.append(partner)
                taken.add(idx)
                taken.add(partner)
            else:
                new_base = _random_base_except(base)
                if new_base != base:
                    seq_list[idx] = new_base
                    changed.append(idx)
                taken.add(idx)
        return ''.join(seq_list), sorted(set(changed))

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------
    def _log_event(
        self,
        event: Dict[str, Any],
        node: InteractiveNode,
    ) -> EventInfo:
        """Create an EventInfo record describing the event before state updates."""
        etype = event.get('type', 'unknown')
        node_name = event.get('node')
        coords: Optional[List[int]] = None
        node_type: Optional[NodeType] = None
        if node_name and node.bulge_graph and node_name in node.bulge_graph.elements:
            coords = list(node.bulge_graph.elements[node_name].positions)
            node_type = classify_node(node_name, coords)
        detail: Dict[str, Any] = {
            'node': node_name,
            'node_type': node_type.value if node_type else None,
            'coords_before': coords,
        }
        desc_parts: List[str] = []

        if etype in ('insert_loop_base', 'delete_loop_base'):
            pos = event.get('pos')
            detail['index'] = pos
            detail['index_1b'] = None if pos is None else pos + 1
            left_col = node.col_map[pos - 1] if pos is not None and pos - 1 >= 0 and pos - 1 < len(node.col_map) else None
            right_col = node.col_map[pos] if pos is not None and pos < len(node.col_map) else None
            detail['left_col'] = left_col
            detail['right_col'] = right_col
            if etype == 'insert_loop_base':
                desc_parts.append(
                    f"Insert loop base at index {pos + 1 if pos is not None else '?'} "
                    f"between column {left_col} and {right_col}"
                )
            else:
                removed_col = node.col_map[pos] if pos is not None and 0 <= pos < len(node.col_map) else None
                detail['removed_col'] = removed_col
                desc_parts.append(
                    f"Delete loop base at index {pos + 1 if pos is not None else '?'} "
                    f"(column {removed_col})"
                )
        elif etype in ('insert_stem_pair', 'delete_stem_pair'):
            left_pos = event.get('left_pos')
            right_pos = event.get('right_pos')
            detail['left_pos'] = left_pos
            detail['right_pos'] = right_pos
            detail['left_pos_1b'] = None if left_pos is None else left_pos + 1
            detail['right_pos_1b'] = None if right_pos is None else right_pos + 1
            left_col = node.col_map[left_pos] if left_pos is not None and 0 <= left_pos < len(node.col_map) else None
            right_col = node.col_map[right_pos] if right_pos is not None and 0 <= right_pos < len(node.col_map) else None
            detail['left_col'] = left_col
            detail['right_col'] = right_col
            if etype == 'insert_stem_pair':
                desc_parts.append(
                    f"Insert stem pair at indices ({left_pos + 1 if left_pos is not None else '?'}, "
                    f"{right_pos + 1 if right_pos is not None else '?'})"
                )
            else:
                desc_parts.append(
                    f"Delete stem pair at indices ({left_pos + 1 if left_pos is not None else '?'}, "
                    f"{right_pos + 1 if right_pos is not None else '?'})"
                )
                detail['removed_cols'] = (
                    node.col_map[left_pos] if left_pos is not None and 0 <= left_pos < len(node.col_map) else None,
                    node.col_map[right_pos] if right_pos is not None and 0 <= right_pos < len(node.col_map) else None,
                )
        else:
            desc_parts.append(etype)

        if node_type:
            desc_parts.append(f"on {node_type.value} node {node_name}")
        elif node_name:
            desc_parts.append(f"on node {node_name}")

        return EventInfo(event_type=etype, description=' '.join(desc_parts), detail=detail)

    def _update_col_map(self, event: Dict[str, Any], node: InteractiveNode) -> None:
        """Mutate the node's column map to keep alignment indices consistent."""
        etype = event.get('type')
        if etype == 'insert_loop_base':
            pos = event.get('pos', 0)
            pos = max(0, min(pos, len(node.col_map)))
            node.col_map.insert(pos, self.next_column_id)
            self.next_column_id += 1
        elif etype == 'delete_loop_base':
            pos = event.get('pos')
            if pos is not None and 0 <= pos < len(node.col_map):
                node.col_map.pop(pos)
        elif etype == 'insert_stem_pair':
            left_pos = event.get('left_pos', 0)
            right_pos = event.get('right_pos', 0)
            right_index = max(0, min(right_pos, len(node.col_map)))
            node.col_map.insert(right_index, self.next_column_id)
            self.next_column_id += 1
            adj_left = left_pos if left_pos < right_pos else left_pos + 1
            adj_left = max(0, min(adj_left, len(node.col_map)))
            node.col_map.insert(adj_left, self.next_column_id)
            self.next_column_id += 1
        elif etype == 'delete_stem_pair':
            left_pos = event.get('left_pos')
            right_pos = event.get('right_pos')
            positions: List[int] = []
            if left_pos is not None:
                positions.append(left_pos)
            if right_pos is not None:
                positions.append(right_pos)
            for idx in sorted(set(positions), reverse=True):
                if 0 <= idx < len(node.col_map):
                    node.col_map.pop(idx)

    # ------------------------------------------------------------------
    # Branch generation
    # ------------------------------------------------------------------
    def generate_branch(
        self,
        parent: InteractiveNode,
        branch_suffix: str,
        root_conserved_cols: set[int],
        root_conserved_pair_list: Optional[List[Tuple[int, int]]],
    ) -> BranchResult:
        node = InteractiveNode(
            path=(parent.path + branch_suffix),
            sequence=parent.sequence,
            structure=parent.structure,
            bulge_graph=deepcopy_bulge_graph(parent.bulge_graph),
            col_map=list(parent.col_map),
        )
        parent_sequence = parent.sequence
        parent_structure = parent.structure

        engine = AlignmentMutationEngine(self.args)
        engine.set_is_conserved_index(lambda i: (0 <= i < len(node.col_map) and node.col_map[i] in root_conserved_cols))

        protected_spans: List[Tuple[int, int]] = []
        paired_root_cols: set[int] = set()
        if root_conserved_pair_list:
            for col_a, col_b in root_conserved_pair_list:
                a, b = (col_a, col_b) if col_a <= col_b else (col_b, col_a)
                protected_spans.append((a, b))
                paired_root_cols.add(col_a)
                paired_root_cols.add(col_b)
        engine.set_is_protected_index(
            lambda i: (0 <= i < len(node.col_map) and any(lo <= node.col_map[i] <= hi for (lo, hi) in protected_spans))
        )

        events: List[EventInfo] = []

        def event_callback(evt: Dict[str, Any]) -> None:
            info = self._log_event(evt, node)
            events.append(info)
            self._update_col_map(evt, node)

        engine.set_event_callback(event_callback)

        sampled = engine.sample_modifications(node.bulge_graph)
        if self.args.mod_normalization:
            factor = max(1.0, len(node.sequence) / self.args.normalization_len)
            sampled = SampledModifications(
                n_stem_indels=int(sampled.n_stem_indels * factor),
                n_hloop_indels=int(sampled.n_hloop_indels * factor),
                n_iloop_indels=int(sampled.n_iloop_indels * factor),
                n_bulge_indels=int(sampled.n_bulge_indels * factor),
                n_mloop_indels=int(sampled.n_mloop_indels * factor),
            )

        for _ in range(sampled.n_stem_indels):
            node.sequence, node.structure = engine._modify_stems(node.sequence, node.structure, node.bulge_graph)
        for _ in range(sampled.n_hloop_indels):
            node.sequence, node.structure = engine._modify_loops(node.sequence, node.structure, node.bulge_graph, NodeType.HAIRPIN)
        for _ in range(sampled.n_iloop_indels):
            node.sequence, node.structure = engine._modify_loops(node.sequence, node.structure, node.bulge_graph, NodeType.INTERNAL)
        for _ in range(sampled.n_bulge_indels):
            node.sequence, node.structure = engine._modify_loops(node.sequence, node.structure, node.bulge_graph, NodeType.BULGE)
        for _ in range(sampled.n_mloop_indels):
            node.sequence, node.structure = engine._modify_loops(node.sequence, node.structure, node.bulge_graph, NodeType.MULTI)

        conserved_indices_now: set[int] = set()
        col_to_idx = {col_id: idx for idx, col_id in enumerate(node.col_map)}
        for idx, col_id in enumerate(node.col_map):
            if col_id in root_conserved_cols:
                conserved_indices_now.add(idx)
        pair_override: Dict[int, int] = {}
        if root_conserved_pair_list:
            for col_a, col_b in root_conserved_pair_list:
                ia = col_to_idx.get(col_a)
                ib = col_to_idx.get(col_b)
                if ia is not None and ib is not None:
                    pair_override[ia] = ib
                    pair_override[ib] = ia

        node.sequence, substitution_positions = self._apply_substitutions(
            node.sequence, node.structure, conserved_indices_now, pair_override=pair_override
        )
        if pair_override:
            seq_list = list(node.sequence)
            visited = set()
            for a, b in list(pair_override.items()):
                if a in visited or b in visited:
                    continue
                if 0 <= a < len(seq_list) and 0 <= b < len(seq_list):
                    seq_list[b] = _complement(seq_list[a])
                    visited.add(a)
                    visited.add(b)
            node.sequence = ''.join(seq_list)

        constraints_pairs: List[Tuple[int, int]] = []
        constraints_unpaired: List[int] = []
        col_to_idx = {col_id: idx for idx, col_id in enumerate(node.col_map)}
        if root_conserved_pair_list:
            for col_a, col_b in root_conserved_pair_list:
                ia = col_to_idx.get(col_a)
                ib = col_to_idx.get(col_b)
                if ia is not None and ib is not None and 0 <= ia < len(node.sequence) and 0 <= ib < len(node.sequence):
                    constraints_pairs.append((ia, ib))
        conserved_single_cols = [col for col in root_conserved_cols if col not in paired_root_cols]
        for col in conserved_single_cols:
            idx = col_to_idx.get(col)
            if idx is not None and 0 <= idx < len(node.sequence):
                constraints_unpaired.append(idx)

        node.structure = self.rna_generator.fold_rna(
            node.sequence,
            constraints_pairs=constraints_pairs,
            constraints_unpaired=constraints_unpaired if constraints_unpaired else None,
        )

        if root_conserved_pair_list:
            s_list = list(node.structure)
            for col_a, col_b in root_conserved_pair_list:
                ia = col_to_idx.get(col_a)
                ib = col_to_idx.get(col_b)
                if ia is not None and ib is not None and 0 <= ia < len(s_list) and 0 <= ib < len(s_list):
                    a, b = (ia, ib) if ia < ib else (ib, ia)
                    s_list[a] = '('
                    s_list[b] = ')'
            node.structure = ''.join(s_list)
        if conserved_single_cols:
            s_list = list(node.structure)
            pair_map_current = _pair_map_from_structure(node.structure)
            for col in conserved_single_cols:
                idx = col_to_idx.get(col)
                if idx is None or not (0 <= idx < len(s_list)):
                    continue
                ch = s_list[idx]
                if ch in '()':
                    partner = pair_map_current.get(idx)
                    if partner is not None and 0 <= partner < len(s_list):
                        s_list[partner] = '.'
                s_list[idx] = '.'
            node.structure = ''.join(s_list)

        node.bulge_graph = self.bulge_parser.parse_structure(node.structure)

        substitution_columns = [node.col_map[pos] for pos in substitution_positions if 0 <= pos < len(node.col_map)]

        mod_counts = engine.type_mod_counts
        action_counts = engine.action_counts

        return BranchResult(
            node=node,
            sampled_mods=sampled,
            mod_counts=mod_counts,
            action_counts=action_counts,
            events=events,
            substitution_positions=substitution_positions,
            substitution_columns=substitution_columns,
            parent_sequence=parent_sequence,
            parent_structure=parent_structure,
            parent_col_map=list(parent.col_map),
        )


def format_sampled_mods(sampled: SampledModifications) -> str:
    return (
        f"stems={sampled.n_stem_indels}, hairpins={sampled.n_hloop_indels}, "
        f"internal={sampled.n_iloop_indels}, bulge={sampled.n_bulge_indels}, "
        f"multi={sampled.n_mloop_indels}"
    )


def format_action_counts(counts: ActionCounts) -> str:
    return (
        f"insertions={counts.total_insertions}, deletions={counts.total_deletions} | "
        f"stem(+/−)={counts.stem_insertions}/{counts.stem_deletions}, hairpin={counts.hloop_insertions}/{counts.hloop_deletions}, "
        f"internal={counts.iloop_insertions}/{counts.iloop_deletions}, bulge={counts.bulge_insertions}/{counts.bulge_deletions}, "
        f"multi={counts.mloop_insertions}/{counts.mloop_deletions}"
    )


def _build_char_map(text: str, columns: List[int]) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    limit = min(len(text), len(columns))
    for idx in range(limit):
        mapping[columns[idx]] = text[idx]
    return mapping


def _merge_column_order(parent_cols: List[int], child_cols: List[int]) -> List[int]:
    order = list(parent_cols)
    position = {col: idx for idx, col in enumerate(order)}

    for idx, col in enumerate(child_cols):
        if col in position:
            continue
        left_pos = None
        for j in range(idx - 1, -1, -1):
            prev_col = child_cols[j]
            if prev_col in position:
                left_pos = position[prev_col]
                break
        right_pos = None
        for j in range(idx + 1, len(child_cols)):
            next_col = child_cols[j]
            if next_col in position:
                right_pos = position[next_col]
                break
        if left_pos is not None:
            insert_at = left_pos + 1
        elif right_pos is not None:
            insert_at = right_pos
        else:
            insert_at = len(order)
        order.insert(insert_at, col)
        # Update cached positions from insert point onwards
        for k in range(insert_at, len(order)):
            position[order[k]] = k
    return order


def _render_alignment(
    order: List[int],
    parent_map: Dict[int, str],
    child_map: Dict[int, str],
    highlight_cols: set[int],
    line_width: int,
) -> List[str]:
    lines: List[str] = []
    for start in range(0, len(order), line_width):
        segment_cols = order[start:start + line_width]
        parent_line = ''.join(parent_map.get(col, '-') for col in segment_cols)
        child_line = ''.join(child_map.get(col, '-') for col in segment_cols)
        marker_line = ''.join('^' if col in highlight_cols else ' ' for col in segment_cols)
        lines.extend([parent_line, child_line, marker_line, ""])
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def display_branch(index: int, result: BranchResult, line_width: int) -> None:
    node = result.node
    print(f"\nBranch {index}: path '{node.path or 'root'}'")
    print(f"  Sampled modification counts: {format_sampled_mods(result.sampled_mods)}")
    print(f"  Actual actions summary:    {format_action_counts(result.action_counts)}")
    if not result.events:
        print("  No indel events applied in this cycle.")
    else:
        print("  Events:")
        for ev in result.events:
            print(f"    - {ev.description}")
    if result.substitution_positions:
        subs_fmt = ', '.join(
            f"idx {pos + 1} (column {col})" for pos, col in zip(result.substitution_positions, result.substitution_columns)
        )
        print(f"  Substitutions at: {subs_fmt}")
    else:
        print("  No substitutions triggered this cycle.")
    print(f"  Result length: {len(node.sequence)} nt (Δ {len(node.sequence) - len(result.parent_sequence)})")
    print("  Sequence:")
    print(textwrap.indent(wrap_text(node.sequence), prefix="    "))
    print("  Structure:")
    print(textwrap.indent(wrap_text(node.structure), prefix="    "))

    parent_seq_map = _build_char_map(result.parent_sequence, result.parent_col_map)
    child_seq_map = _build_char_map(node.sequence, node.col_map)
    parent_struct_map = _build_char_map(result.parent_structure, result.parent_col_map)
    child_struct_map = _build_char_map(node.structure, node.col_map)

    column_order = _merge_column_order(result.parent_col_map, node.col_map)

    sequence_diff_cols = {
        col for col in column_order
        if parent_seq_map.get(col, '-') != child_seq_map.get(col, '-')
    }
    sequence_diff_cols.update(result.substitution_columns)

    structure_diff_cols = {
        col for col in column_order
        if parent_struct_map.get(col, '-') != child_struct_map.get(col, '-')
    }

    print("Sequence alignment vs parent:")
    seq_lines = _render_alignment(column_order, parent_seq_map, child_seq_map, sequence_diff_cols, line_width)
    for line in seq_lines:
        print(line)
    print("\nStructure alignment vs parent:")
    struct_lines = _render_alignment(column_order, parent_struct_map, child_struct_map, structure_diff_cols, line_width)
    for line in struct_lines:
        print(line)


def run_interactive(args: argparse.Namespace) -> None:
    # Seed with system entropy for variability; users can control via PYTHONHASHSEED
    random.seed()
    runner = RealTimeAlignmentRunner(args)

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    root = runner.create_root()
    conserved_cols, conserved_pairs = runner.select_conserved_sites(root.structure)

    print("Initial sequence (root):")
    print(textwrap.indent(wrap_text(root.sequence), prefix="  "))
    print("Initial structure:")
    print(textwrap.indent(wrap_text(root.structure), prefix="  "))
    print(f"Length: {len(root.sequence)} nt")
    print(f"Conserved fraction target: {args.f_conserved_sites:.3f} → {len(conserved_cols)} columns flagged")
    if conserved_pairs:
        pairs_fmt = ', '.join(f"({a + 1},{b + 1})" for a, b in conserved_pairs)
        print(f"Conserved pairs (1-based indices on root): {pairs_fmt}")
    else:
        print("No conserved pairs selected for this run.")

    history: List[Dict[str, Any]] = []
    current = root

    cycle = 1
    while True:
        response = input(f"\nReady for cycle {cycle}? Press Enter to continue or 'q' to stop: ")
        if response.strip().lower() in {'q', 'quit', 'exit'}:
            print("Stopping interactive session.")
            break

        branches = [
            runner.generate_branch(current, '0', conserved_cols, conserved_pairs),
            runner.generate_branch(current, '1', conserved_cols, conserved_pairs),
        ]

        for idx, branch in enumerate(branches, start=1):
            display_branch(idx, branch, args.line_width)

        choice = None
        while choice not in {'1', '2'}:
            choice = input("Select branch to continue with (1/2): ").strip()
        chosen_idx = int(choice) - 1
        chosen_branch = branches[chosen_idx]

        history.append({
            'cycle': cycle,
            'chosen_branch': chosen_idx + 1,
            'path': chosen_branch.node.path,
            'events': [ev.description for ev in chosen_branch.events],
            'substitutions': [
                {
                    'index': pos + 1,
                    'column': col,
                }
                for pos, col in zip(chosen_branch.substitution_positions, chosen_branch.substitution_columns)
            ],
            'length': len(chosen_branch.node.sequence),
        })

        current = chosen_branch.node
        cycle += 1

    print("\nSession summary:")
    for entry in history:
        print(f"  Cycle {entry['cycle']}: chose branch {entry['chosen_branch']} (path {entry['path'] or 'root'})")
        if entry['events']:
            for desc in entry['events']:
                print(f"    - {desc}")
        else:
            print("    - No indel events")
        if entry['substitutions']:
            subs_fmt = ', '.join(f"idx {item['index']} (column {item['column']})" for item in entry['substitutions'])
            print(f"    - Substitutions: {subs_fmt}")
        print(f"    - Sequence length now {entry['length']} nt")

    print("\nFinal sequence:")
    print(textwrap.indent(wrap_text(current.sequence), prefix="  "))
    print("Final structure:")
    print(textwrap.indent(wrap_text(current.structure), prefix="  "))
    print(f"Final length: {len(current.sequence)} nt")


def main() -> int:
    args = parse_generator_arguments()
    try:
        run_interactive(args)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 1
    except Exception as exc:  # pragma: no cover - interactive debugging helper
        print(f"Error: {exc}", file=sys.stderr)
        logger.exception("Interactive session failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
