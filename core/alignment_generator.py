"""
Alignment dataset generator: simulates evolution from an ancestral sequence
over a binary tree, applying substitutions and indels while conserving a
fraction of base pairs at each node, and outputs an alignment of leaf sequences.
"""

import copy
import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable

from .models import (
    AlignmentLeaf,
    AlignmentResult,
    BulgeGraph,
)
from .rna_generator import RnaGenerator, BulgeGraphParser
from .modification_engine import ModificationEngine, NodeType, SampledModifications

logger = logging.getLogger(__name__)


def _pair_map_from_structure(structure: str) -> Dict[int, int]:
    """Build a 0-based index map from base-paired positions to their partners."""
    stack = []
    pair_map: Dict[int, int] = {}
    for i, ch in enumerate(structure):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            if stack:
                j = stack.pop()
                pair_map[i] = j
                pair_map[j] = i
    return pair_map


def _random_base_except(current: str) -> str:
    bases = ['A', 'C', 'G', 'U']
    choices = [b for b in bases if b != current]
    return random.choice(choices) if choices else current


def _complement(base: str) -> str:
    comp = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
    return comp.get(base, base)


@dataclass
class _EvolutionNode:
    path: str
    sequence: str
    structure: str
    bulge_graph: BulgeGraph
    col_map: List[int]            # maps local sequence indices -> global column IDs


class AlignmentMutationEngine(ModificationEngine):
    """Extension of ModificationEngine that emits alignment-aware events and
    avoids deleting conserved stem pairs when requested.
    """

    def __init__(self, args):
        super().__init__(args)
        self.event_callback: Optional[Callable[[Dict], None]] = None
        self._conserved_pairs: Optional[set[int]] = None  # 0-based indices in sequence

    def set_event_callback(self, cb: Optional[Callable[[Dict], None]]):
        self.event_callback = cb

    def set_conserved_pairs(self, conserved_zero_based: Optional[set[int]]):
        self._conserved_pairs = conserved_zero_based or set()

    def _emit(self, event: Dict):
        if self.event_callback:
            try:
                self.event_callback(event)
            except Exception as e:
                logger.debug(f"Event callback error ignored: {e}")

    # Override stem/loop ops to emit events and (for stems) avoid conserved deletions
    def _insert_stem_pair(self, sequence: str, structure: str, node_name: str,
                          coords: List[int], bulge_graph: BulgeGraph) -> Tuple[str, str]:
        if len(coords) < 2:
            return sequence, structure

        sorted_coords = sorted(coords)
        n = len(sorted_coords)
        left_half = sorted_coords[: n // 2]
        right_half = sorted_coords[n // 2 :]

        left_insert_options = []
        for i in range(len(left_half)):
            if i == 0:
                left_insert_options.append(left_half[i] - 1)
            left_insert_options.append(left_half[i])
        left_pos = random.choice(left_insert_options)

        left_gap_index = 0
        for i, pos in enumerate(left_half):
            if left_pos <= pos:
                left_gap_index = i
                break
        else:
            left_gap_index = len(left_half)

        if left_gap_index == 0:
            right_pos = right_half[-1]
        elif left_gap_index == len(left_half):
            right_pos = right_half[0] - 1
        else:
            right_mirror_index = len(right_half) - left_gap_index
            right_pos = right_half[right_mirror_index - 1] if right_mirror_index > 0 else right_half[0] - 1

        complement_pairs = [('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G')]
        base_left, base_right = random.choice(complement_pairs)

        # Emit event before mutating
        self._emit({
            'type': 'insert_stem_pair',
            'node': node_name,
            'left_pos': left_pos,
            'right_pos': right_pos,
        })

        # Apply insertion, respecting index order
        if left_pos < right_pos:
            sequence = sequence[:right_pos] + base_right + sequence[right_pos:]
            structure = structure[:right_pos] + ')' + structure[right_pos:]
            sequence = sequence[:left_pos] + base_left + sequence[left_pos:]
            structure = structure[:left_pos] + '(' + structure[left_pos:]
        else:
            sequence = sequence[:left_pos] + base_left + sequence[left_pos:]
            structure = structure[:left_pos] + '(' + structure[left_pos:]
            sequence = sequence[:right_pos] + base_right + sequence[right_pos:]
            structure = structure[:right_pos] + ')' + structure[right_pos:]

        from .bulge_graph_updater import BulgeGraphUpdater
        BulgeGraphUpdater.insert_stem_pair(bulge_graph, node_name, left_pos, right_pos)
        return sequence, structure

    def _delete_stem_pair(self, sequence: str, structure: str, node_name: str,
                          coords: List[int], bulge_graph: BulgeGraph) -> Tuple[str, str]:
        if len(coords) < 4:
            return sequence, structure

        sorted_coords = sorted(coords)
        n = len(sorted_coords)
        left_half = sorted_coords[: n // 2]
        right_half = sorted_coords[n // 2 :]

        # Build list of deletable left indices avoiding conserved pairs when configured
        candidates: List[int] = []
        for idx in range(len(left_half)):
            left_pos_1b = left_half[idx]
            right_pos_1b = right_half[len(right_half) - 1 - idx]
            left0 = left_pos_1b - 1
            right0 = right_pos_1b - 1
            if self._conserved_pairs and (left0 in self._conserved_pairs or right0 in self._conserved_pairs):
                continue
            candidates.append(idx)

        if not candidates:
            # No allowed deletion; do nothing
            return sequence, structure

        left_delete_idx = random.choice(candidates)
        left_pos = left_half[left_delete_idx] - 1
        right_delete_idx = len(right_half) - 1 - left_delete_idx
        right_pos = right_half[right_delete_idx] - 1

        # Emit event before mutating
        self._emit({
            'type': 'delete_stem_pair',
            'node': node_name,
            'left_pos': left_pos,
            'right_pos': right_pos,
        })

        if right_pos < len(sequence) and left_pos < len(sequence) and right_pos > left_pos:
            sequence = sequence[:right_pos] + sequence[right_pos+1:]
            structure = structure[:right_pos] + structure[right_pos+1:]
            sequence = sequence[:left_pos] + sequence[left_pos+1:]
            structure = structure[:left_pos] + structure[left_pos+1:]
        elif left_pos < len(sequence) and right_pos < len(sequence) and left_pos > right_pos:
            sequence = sequence[:left_pos] + sequence[left_pos+1:]
            structure = structure[:left_pos] + structure[left_pos+1:]
            sequence = sequence[:right_pos] + sequence[right_pos+1:]
            structure = structure[:right_pos] + structure[right_pos+1:]

        from .bulge_graph_updater import BulgeGraphUpdater
        BulgeGraphUpdater.delete_stem_pair(bulge_graph, node_name, left_pos, right_pos)
        return sequence, structure

    def _insert_loop_base(self, sequence: str, structure: str, node_name: str,
                          coords: List[int], bulge_graph: BulgeGraph) -> Tuple[str, str]:
        if not coords:
            return sequence, structure
        pos = random.choice(coords) - 1
        base = random.choice(['A', 'C', 'G', 'U'])
        # Emit event
        self._emit({'type': 'insert_loop_base', 'node': node_name, 'pos': pos})
        sequence = sequence[:pos] + base + sequence[pos:]
        structure = structure[:pos] + '.' + structure[pos:]
        from .bulge_graph_updater import BulgeGraphUpdater
        BulgeGraphUpdater.insert_loop_base(bulge_graph, node_name, pos)
        return sequence, structure

    def _delete_loop_base(self, sequence: str, structure: str, node_name: str,
                          coords: List[int], node_type: NodeType, min_size: int, bulge_graph: BulgeGraph) -> Tuple[str, str]:
        if not coords:
            return sequence, structure

        # Size validations same as base class
        if node_type == NodeType.INTERNAL and len(coords) >= 4:
            side1_size = coords[1] - coords[0] + 1
            side2_size = coords[3] - coords[2] + 1
            if min(side1_size, side2_size) <= min_size:
                return sequence, structure
        else:
            if len(coords) >= 2:
                current_size = coords[1] - coords[0] + 1
                if current_size <= min_size:
                    return sequence, structure

        pos = random.choice(coords) - 1
        # Emit event
        self._emit({'type': 'delete_loop_base', 'node': node_name, 'pos': pos})
        if pos < len(sequence):
            sequence = sequence[:pos] + sequence[pos+1:]
            structure = structure[:pos] + structure[pos+1:]
            from .bulge_graph_updater import BulgeGraphUpdater
            BulgeGraphUpdater.delete_loop_base(bulge_graph, node_name, pos)
        return sequence, structure


class AlignmentDatasetGenerator:
    """Generates multiple alignments according to the requested parameters."""

    def __init__(self, args):
        self.args = args
        self.rna_generator = RnaGenerator()
        self.bulge_parser = BulgeGraphParser()

    def _deepcopy_graph(self, bg: BulgeGraph) -> BulgeGraph:
        return BulgeGraph(elements={
            name: type(node)(positions=list(node.positions), start=node.start, end=node.end)
            for name, node in bg.elements.items()
        })

    def _choose_length(self) -> int:
        return self.rna_generator.choose_sequence_length(
            self.args.seq_len_distribution,
            self.args.seq_min_len,
            self.args.seq_max_len,
            self.args.seq_len_mean,
            self.args.seq_len_sd,
        )

    def _init_root(self) -> _EvolutionNode:
        length = self._choose_length()
        sequence = self.rna_generator.generate_random_sequence(length)
        structure = self.rna_generator.fold_rna(sequence)
        bulge = self.bulge_parser.parse_structure(structure)
        col_map = list(range(length))
        return _EvolutionNode(path="", sequence=sequence, structure=structure, bulge_graph=bulge, col_map=col_map)

    def _select_conserved_pairs(self, structure: str) -> set[int]:
        pair_map = _pair_map_from_structure(structure)
        # build unique pairs using left indices
        pairs = []
        used = set()
        for i, j in sorted(pair_map.items()):
            if i < j and (i, j) not in used and (j, i) not in used:
                pairs.append((i, j))
                used.add((i, j))
        if not pairs:
            return set()
        k = max(0, int(round(len(pairs) * self.args.f_conserved_sites)))
        chosen = set(random.sample(pairs, k)) if k > 0 else set()
        conserved: set[int] = set()
        for i, j in chosen:
            conserved.add(i)
            conserved.add(j)
        return conserved

    def _apply_substitutions(self, sequence: str, structure: str, conserved_pairs: set[int]) -> str:
        if not sequence:
            return sequence
        n = len(sequence)
        n_subs = max(0, int(round(n * self.args.f_substitution_rate)))
        if n_subs == 0:
            return sequence
        pair_map = _pair_map_from_structure(structure)
        indices = list(range(n))
        random.shuffle(indices)
        taken = set()
        seq_list = list(sequence)
        for idx in indices:
            if len(taken) >= n_subs:
                break
            if idx in taken:
                continue
            base = seq_list[idx]
            if idx in pair_map and idx in conserved_pairs:
                partner = pair_map[idx]
                if partner in taken:
                    continue
                # mutate idx and set partner to complement
                new_base = _random_base_except(base)
                seq_list[idx] = new_base
                seq_list[partner] = _complement(new_base)
                taken.add(idx)
                taken.add(partner)
            else:
                seq_list[idx] = _random_base_except(base)
                taken.add(idx)
        return ''.join(seq_list)

    def _event_cb_builder(self, node: _EvolutionNode, column_order: List[int], next_col_id_ref: List[int]):
        # next_col_id_ref is a single-item list used as a mutable integer holder
        def cb(event: Dict):
            etype = event.get('type')
            if etype == 'insert_loop_base':
                pos = event['pos']
                # insert new global column id into column_order after the left neighbor of this pos
                new_id = next_col_id_ref[0]
                next_col_id_ref[0] += 1
                # compute global insert index
                if pos > 0:
                    left_col = node.col_map[pos - 1]
                    gi = column_order.index(left_col) + 1
                else:
                    gi = 0
                column_order.insert(gi, new_id)
                # update node's col_map
                node.col_map.insert(pos, new_id)
            elif etype == 'delete_loop_base':
                pos = event['pos']
                if 0 <= pos < len(node.col_map):
                    node.col_map.pop(pos)
            elif etype == 'insert_stem_pair':
                left_pos = event['left_pos']
                right_pos = event['right_pos']
                # Insert two new columns using left and right positions independently
                # Right insertion first in global order update analogous to sequence mutation order
                # Right
                new_r = next_col_id_ref[0]
                next_col_id_ref[0] += 1
                if right_pos > 0:
                    left_neighbor = node.col_map[right_pos - 1]
                    gi = column_order.index(left_neighbor) + 1
                else:
                    gi = 0
                column_order.insert(gi, new_r)
                node.col_map.insert(right_pos, new_r)
                # Adjust left_pos if right_pos <= left_pos due to insert
                adj_left_pos = left_pos if left_pos < right_pos else left_pos + 1
                new_l = next_col_id_ref[0]
                next_col_id_ref[0] += 1
                if adj_left_pos > 0:
                    left_neighbor = node.col_map[adj_left_pos - 1]
                    gi2 = column_order.index(left_neighbor) + 1
                else:
                    gi2 = 0
                column_order.insert(gi2, new_l)
                node.col_map.insert(adj_left_pos, new_l)
            elif etype == 'delete_stem_pair':
                left_pos = event['left_pos']
                right_pos = event['right_pos']
                # Delete higher index first from col_map
                first = max(left_pos, right_pos)
                second = min(left_pos, right_pos)
                if 0 <= first < len(node.col_map):
                    node.col_map.pop(first)
                if 0 <= second < len(node.col_map):
                    node.col_map.pop(second)
        return cb

    def _sample_mods(self, engine: ModificationEngine, bulge_graph: BulgeGraph) -> SampledModifications:
        sampled = engine.sample_modifications(bulge_graph)
        if self.args.mod_normalization:
            # Scale counts by sequence length relative to normalization_len
            # Use bulge_graph total nucleotide length approximation via sum of node positions
            # Fallback to sequence length derived from any element count
            # We will simply approximate by using sum of all element lengths in mapping or len of any structure string
            # since we do not have the sequence here, leave as: factor computed later per node length
            # Here we return sampled and scale later when we know sequence length per node
            pass
        return sampled

    def _mutate_child(self, parent: _EvolutionNode, child_path: str, column_order: List[int], next_col_id_ref: List[int], root_conserved_cols: set[int]) -> _EvolutionNode:
        # Copy parent state
        seq = parent.sequence
        struct = parent.structure
        bg = self._deepcopy_graph(parent.bulge_graph)
        node = _EvolutionNode(path=child_path, sequence=seq, structure=struct, bulge_graph=bg, col_map=list(parent.col_map))

        # Map root conserved column ids to current sequence indices for this node
        conserved_indices: set[int] = set()
        for idx, col_id in enumerate(node.col_map):
            if col_id in root_conserved_cols:
                conserved_indices.add(idx)

        # Indels using alignment-aware engine
        engine = AlignmentMutationEngine(self.args)
        engine.set_conserved_pairs(conserved_indices)
        engine.set_event_callback(self._event_cb_builder(node, column_order, next_col_id_ref))

        sampled = self._sample_mods(engine, node.bulge_graph)
        # Apply length-based normalization if requested
        if self.args.mod_normalization:
            factor = max(1.0, len(node.sequence) / self.args.normalization_len)
            sampled = SampledModifications(
                n_stem_indels=int(sampled.n_stem_indels * factor),
                n_hloop_indels=int(sampled.n_hloop_indels * factor),
                n_iloop_indels=int(sampled.n_iloop_indels * factor),
                n_bulge_indels=int(sampled.n_bulge_indels * factor),
                n_mloop_indels=int(sampled.n_mloop_indels * factor),
            )

        # Apply stem modifications
        for _ in range(sampled.n_stem_indels):
            node.sequence, node.structure = engine._modify_stems(node.sequence, node.structure, node.bulge_graph)

        # Apply loop modifications
        for _ in range(sampled.n_hloop_indels):
            node.sequence, node.structure = engine._modify_loops(node.sequence, node.structure, node.bulge_graph, NodeType.HAIRPIN)
        for _ in range(sampled.n_iloop_indels):
            node.sequence, node.structure = engine._modify_loops(node.sequence, node.structure, node.bulge_graph, NodeType.INTERNAL)
        for _ in range(sampled.n_bulge_indels):
            node.sequence, node.structure = engine._modify_loops(node.sequence, node.structure, node.bulge_graph, NodeType.BULGE)
        for _ in range(sampled.n_mloop_indels):
            node.sequence, node.structure = engine._modify_loops(node.sequence, node.structure, node.bulge_graph, NodeType.MULTI)

        # Substitutions with compensatory logic on conserved stems
        node.sequence = self._apply_substitutions(node.sequence, node.structure, conserved_indices)

        # Re-fold to get updated structure for downstream nodes
        node.structure = self.rna_generator.fold_rna(node.sequence)
        node.bulge_graph = self.bulge_parser.parse_structure(node.structure)
        return node

    def _evolve_tree(self, root: _EvolutionNode) -> AlignmentResult:
        """Evolve the tree breadth-first for num_cycles, returning the leaves."""
        # Global column order and next column id tracker
        column_order: List[int] = list(root.col_map)
        next_col_id_ref = [len(root.col_map)]  # mutable holder

        # Determine globally conserved columns from root structure (paired positions subset)
        root_conserved_cols: set[int] = self._select_conserved_pairs(root.structure)

        current_level = [root]
        for depth in range(self.args.num_cycles):
            next_level: List[_EvolutionNode] = []
            for node in current_level:
                left_child = self._mutate_child(node, node.path + '0', column_order, next_col_id_ref, root_conserved_cols)
                right_child = self._mutate_child(node, node.path + '1', column_order, next_col_id_ref, root_conserved_cols)
                next_level.append(left_child)
                next_level.append(right_child)
            current_level = next_level

        # Build aligned leaves strings using final column_order
        aligned_leaves: List[AlignmentLeaf] = []
        for node in current_level:
            # Map from column id -> base for this leaf
            col_to_base: Dict[int, str] = {}
            for idx, col_id in enumerate(node.col_map):
                if 0 <= idx < len(node.sequence):
                    col_to_base[col_id] = node.sequence[idx]
            # Map from column id -> structure char for this leaf
            col_to_struct: Dict[int, str] = {}
            for idx, col_id in enumerate(node.col_map):
                if 0 <= idx < len(node.structure):
                    col_to_struct[col_id] = node.structure[idx]
            # emit aligned sequence following column_order; gap if missing
            chars = []
            sschars = []
            for col_id in column_order:
                chars.append(col_to_base.get(col_id, '-'))
                sschars.append(col_to_struct.get(col_id, '-'))
            aligned_seq = ''.join(chars)
            aligned_ss = ''.join(sschars)
            aligned_leaves.append(AlignmentLeaf(
                leaf_id=node.path if node.path else "root",
                path=node.path,
                sequence=node.sequence,
                structure=node.structure,
                aligned_sequence=aligned_seq,
                aligned_structure=aligned_ss,
            ))

        # Build GC conservation string per column in final order
        gc_chars = ['1' if col_id in root_conserved_cols else '0' for col_id in column_order]
        gc_str = ''.join(gc_chars)

        return AlignmentResult(
            alignment_id=0,  # To be filled by caller
            leaves=aligned_leaves,
            column_count=len(column_order),
            gc_conservation=gc_str,
        )

    def generate_alignments(self) -> List[AlignmentResult]:
        results: List[AlignmentResult] = []
        for aid in range(self.args.num_alignments):
            root = self._init_root()
            res = self._evolve_tree(root)
            # Patch alignment id
            res.alignment_id = aid
            results.append(res)
        return results
