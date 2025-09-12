"""
RNA modification engine for applying structural changes.
"""

import logging
import random
from typing import Dict, List, Tuple, Optional
from scipy.stats import truncnorm

from .models import (
    BulgeGraph, NodeType, ModificationType, ModificationCounts,
    SampledModifications, ActionCounts, classify_node
)

logger = logging.getLogger(__name__)


class ModificationEngine:
    """Handles applying modifications to RNA sequences and structures."""
    
    def __init__(self, args):
        """Initialize the modification engine with configuration."""
        self.args = args
        self.modification_counts = {}
        self.node_actions = {}
        # Track counts by structural element type as modifications are applied
        self.type_mod_counts = ModificationCounts()
        self.action_counts = ActionCounts()
    
    def _format_bulge_graph_summary(self, bulge_graph: BulgeGraph, sequence: str, structure: str) -> str:
        """Format a detailed summary of the bulge graph state with nodes sorted by first coordinate."""
        node_mapping = bulge_graph.node_mapping()
        
        # Sort nodes by their first coordinate for consistent output
        sorted_nodes = []
        for node_name, coords in node_mapping.items():
            first_coord = coords[0] if coords else float('inf')
            sorted_nodes.append((first_coord, node_name, coords))
        
        sorted_nodes.sort(key=lambda x: x[0])
        
        summary_lines = [
            f"Sequence:  {sequence}",
            f"Structure: {structure}",
            f"BulgeGraph Summary ({len(node_mapping)} nodes):"
        ]
        
        for _, node_name, coords in sorted_nodes:
            node_type = classify_node(node_name, coords)
            mod_count = self.modification_counts.get(node_name, 0)
            summary_lines.append(f"  {node_name:<4}: {coords} → {node_type.value} (modified {mod_count} times)")
        
        return "\n".join(summary_lines)

    def sample_modifications(self, bulge_graph: BulgeGraph) -> SampledModifications:
        """Sample modification counts using either absolute numbers or fractions."""

        def _total_length(ntype: NodeType) -> int:
            total = 0
            for _, coords in bulge_graph.get_nodes_by_type(ntype):
                total += self._get_nucleotide_size(ntype, coords)
            return total

        def _fraction_count(mean, sd, mn, mx, total):
            frac = self._sample_truncated_normal_float(mean, sd, mn, mx)
            return int(round(total * frac))

        # Stem sampling
        if self.args.f_stem_indels_mean is not None:
            stem_total = _total_length(NodeType.STEM)
            n_stem = _fraction_count(
                self.args.f_stem_indels_mean,
                self.args.f_stem_indels_sd,
                self.args.f_stem_indels_min,
                self.args.f_stem_indels_max,
                stem_total,
            )
        else:
            n_stem = self._sample_truncated_normal(
                self.args.n_stem_indels_mean,
                self.args.n_stem_indels_sd,
                self.args.n_stem_indels_min,
                self.args.n_stem_indels_max,
            )

        # Hairpin loop sampling
        if self.args.f_hloop_indels_mean is not None:
            total = _total_length(NodeType.HAIRPIN)
            n_hloop = _fraction_count(
                self.args.f_hloop_indels_mean,
                self.args.f_hloop_indels_sd,
                self.args.f_hloop_indels_min,
                self.args.f_hloop_indels_max,
                total,
            )
        else:
            n_hloop = self._sample_truncated_normal(
                self.args.n_hloop_indels_mean,
                self.args.n_hloop_indels_sd,
                self.args.n_hloop_indels_min,
                self.args.n_hloop_indels_max,
            )

        # Internal loop sampling
        if self.args.f_iloop_indels_mean is not None:
            total = _total_length(NodeType.INTERNAL)
            n_iloop = _fraction_count(
                self.args.f_iloop_indels_mean,
                self.args.f_iloop_indels_sd,
                self.args.f_iloop_indels_min,
                self.args.f_iloop_indels_max,
                total,
            )
        else:
            n_iloop = self._sample_truncated_normal(
                self.args.n_iloop_indels_mean,
                self.args.n_iloop_indels_sd,
                self.args.n_iloop_indels_min,
                self.args.n_iloop_indels_max,
            )

        # Bulge loop sampling
        if self.args.f_bulge_indels_mean is not None:
            total = _total_length(NodeType.BULGE)
            n_bulge = _fraction_count(
                self.args.f_bulge_indels_mean,
                self.args.f_bulge_indels_sd,
                self.args.f_bulge_indels_min,
                self.args.f_bulge_indels_max,
                total,
            )
        else:
            n_bulge = self._sample_truncated_normal(
                self.args.n_bulge_indels_mean,
                self.args.n_bulge_indels_sd,
                self.args.n_bulge_indels_min,
                self.args.n_bulge_indels_max,
            )

        # Multi loop sampling
        if self.args.f_mloop_indels_mean is not None:
            total = _total_length(NodeType.MULTI)
            n_mloop = _fraction_count(
                self.args.f_mloop_indels_mean,
                self.args.f_mloop_indels_sd,
                self.args.f_mloop_indels_min,
                self.args.f_mloop_indels_max,
                total,
            )
        else:
            n_mloop = self._sample_truncated_normal(
                self.args.n_mloop_indels_mean,
                self.args.n_mloop_indels_sd,
                self.args.n_mloop_indels_min,
                self.args.n_mloop_indels_max,
            )

        return SampledModifications(
            n_stem_indels=n_stem,
            n_hloop_indels=n_hloop,
            n_iloop_indels=n_iloop,
            n_bulge_indels=n_bulge,
            n_mloop_indels=n_mloop,
        )
    
    def _sample_truncated_normal(self, mean: float, sd: float, min_val: int, max_val: int) -> int:
        """Sample from a truncated normal distribution."""
        if min_val >= max_val:
            return min_val

        a, b = (min_val - mean) / sd, (max_val - mean) / sd
        sample = truncnorm.rvs(a, b, loc=mean, scale=sd)
        return int(round(sample))

    def _sample_truncated_normal_float(self, mean: float, sd: float, min_val: float, max_val: float) -> float:
        """Sample a float from a truncated normal distribution."""
        if min_val >= max_val:
            return min_val

        a, b = (min_val - mean) / sd, (max_val - mean) / sd
        return float(truncnorm.rvs(a, b, loc=mean, scale=sd))
    
    def apply_modifications(self, sequence: str, structure: str, 
                          bulge_graph: BulgeGraph,
                          sampled_mods: SampledModifications) -> Tuple[str, str, ModificationCounts, ActionCounts]:
        """
        Apply modifications to create a positive sample.
        
        Args:
            sequence: Original RNA sequence
            structure: Original dot-bracket structure
            bulge_graph: Parsed bulge graph
            sampled_mods: Sampled modification counts
            
        Returns:
            Tuple of (modified_sequence, modified_structure, modification_counts)
        """
        pos_seq = sequence
        pos_struct = structure
        self.modification_counts = {}
        self.node_actions = {}
        # Reset type-based modification counters
        self.type_mod_counts = ModificationCounts()
        self.action_counts = ActionCounts()
        
        # Normalize modification counts if enabled
        if self.args.mod_normalization:
            factor = max(1.0, len(sequence) / self.args.normalization_len)
            sampled_mods = SampledModifications(
                n_stem_indels=int(sampled_mods.n_stem_indels * factor),
                n_hloop_indels=int(sampled_mods.n_hloop_indels * factor),
                n_iloop_indels=int(sampled_mods.n_iloop_indels * factor),
                n_bulge_indels=int(sampled_mods.n_bulge_indels * factor),
                n_mloop_indels=int(sampled_mods.n_mloop_indels * factor)
            )
        
        logger.debug(f"Applying modifications: {sampled_mods}")
        
        # Show initial bulge graph state
        logger.debug("=== Initial bulge graph state ===")
        logger.debug(self._format_bulge_graph_summary(bulge_graph, pos_seq, pos_struct))
        
        # Apply stem modifications
        logger.debug(f"=== Starting stem modifications ({sampled_mods.n_stem_indels} planned) ===")
        for i in range(sampled_mods.n_stem_indels):
            logger.debug(f"--- Stem modification {i + 1}/{sampled_mods.n_stem_indels} ---")
            logger.debug(f"Before stem modification {i + 1}:")
            logger.debug(self._format_bulge_graph_summary(bulge_graph, pos_seq, pos_struct))
            
            new_seq, new_struct = self._modify_stems(pos_seq, pos_struct, bulge_graph)

            # Bulge graph updated in place
            pos_seq, pos_struct = new_seq, new_struct
            
            logger.debug(f"After stem modification {i + 1}:")
            logger.debug(self._format_bulge_graph_summary(bulge_graph, pos_seq, pos_struct))
        
        # Apply loop modifications
        logger.debug(f"=== Starting hairpin loop modifications ({sampled_mods.n_hloop_indels} planned) ===")
        for i in range(sampled_mods.n_hloop_indels):
            logger.debug(f"--- Hairpin modification {i + 1}/{sampled_mods.n_hloop_indels} ---")
            logger.debug(f"Before hairpin modification {i + 1}:")
            logger.debug(self._format_bulge_graph_summary(bulge_graph, pos_seq, pos_struct))
            
            new_seq, new_struct = self._modify_loops(pos_seq, pos_struct, bulge_graph, NodeType.HAIRPIN)

            # Bulge graph updated in place
            pos_seq, pos_struct = new_seq, new_struct
            
            logger.debug(f"After hairpin modification {i + 1}:")
            logger.debug(self._format_bulge_graph_summary(bulge_graph, pos_seq, pos_struct))
        
        logger.debug(f"=== Starting internal loop modifications ({sampled_mods.n_iloop_indels} planned) ===")
        for i in range(sampled_mods.n_iloop_indels):
            logger.debug(f"--- Internal loop modification {i + 1}/{sampled_mods.n_iloop_indels} ---")
            logger.debug(f"Before internal loop modification {i + 1}:")
            logger.debug(self._format_bulge_graph_summary(bulge_graph, pos_seq, pos_struct))
            
            new_seq, new_struct = self._modify_loops(pos_seq, pos_struct, bulge_graph, NodeType.INTERNAL)

            # Bulge graph updated in place
            pos_seq, pos_struct = new_seq, new_struct
            
            logger.debug(f"After internal loop modification {i + 1}:")
            logger.debug(self._format_bulge_graph_summary(bulge_graph, pos_seq, pos_struct))
        
        logger.debug(f"=== Starting bulge modifications ({sampled_mods.n_bulge_indels} planned) ===")
        for i in range(sampled_mods.n_bulge_indels):
            logger.debug(f"--- Bulge modification {i + 1}/{sampled_mods.n_bulge_indels} ---")
            logger.debug(f"Before bulge modification {i + 1}:")
            logger.debug(self._format_bulge_graph_summary(bulge_graph, pos_seq, pos_struct))
            
            new_seq, new_struct = self._modify_loops(pos_seq, pos_struct, bulge_graph, NodeType.BULGE)

            # Bulge graph updated in place
            pos_seq, pos_struct = new_seq, new_struct
            
            logger.debug(f"After bulge modification {i + 1}:")
            logger.debug(self._format_bulge_graph_summary(bulge_graph, pos_seq, pos_struct))
        
        logger.debug(f"=== Starting multi loop modifications ({sampled_mods.n_mloop_indels} planned) ===")
        for i in range(sampled_mods.n_mloop_indels):
            logger.debug(f"--- Multi loop modification {i + 1}/{sampled_mods.n_mloop_indels} ---")
            logger.debug(f"Before multi loop modification {i + 1}:")
            logger.debug(self._format_bulge_graph_summary(bulge_graph, pos_seq, pos_struct))
            
            new_seq, new_struct = self._modify_loops(pos_seq, pos_struct, bulge_graph, NodeType.MULTI)

            # Bulge graph updated in place
            pos_seq, pos_struct = new_seq, new_struct
            
            logger.debug(f"After multi loop modification {i + 1}:")
            logger.debug(self._format_bulge_graph_summary(bulge_graph, pos_seq, pos_struct))
        
        # Show final bulge graph state
        logger.debug("=== Final bulge graph state ===")
        logger.debug(self._format_bulge_graph_summary(bulge_graph, pos_seq, pos_struct))
        logger.debug(f"Length changed from {len(sequence)} to {len(pos_seq)}")
        
        # Calculate final modification counts
        mod_counts = self._calculate_modification_counts(bulge_graph)
        action_counts = self._calculate_action_counts()
        
        logger.debug("=== Modification summary ===")
        logger.debug(f"Total modifications: {mod_counts.total}")
        logger.debug(f"Stem modifications: {mod_counts.stem}")
        logger.debug(f"Hairpin modifications: {mod_counts.hloop}")
        logger.debug(f"Internal loop modifications: {mod_counts.iloop}")
        logger.debug(f"Bulge modifications: {mod_counts.bulge}")
        logger.debug(f"Multi loop modifications: {mod_counts.mloop}")

        return pos_seq, pos_struct, mod_counts, action_counts

    def _modify_stems(
        self, sequence: str, structure: str, bulge_graph: BulgeGraph
    ) -> Tuple[str, str]:
        """Apply a single insertion or deletion on a random stem."""
        eligible = self._get_eligible_nodes(
            bulge_graph,
            NodeType.STEM,
            self.args.stem_min_size,
            self.args.stem_max_size,
            self.args.same_stem_max_n_mod,
        )

        if not eligible:
            return sequence, structure

        node_name, coords = random.choice(eligible)
        action = self._choose_action(
            node_name,
            coords,
            NodeType.STEM,
            self.args.stem_min_size,
            self.args.stem_max_size,
        )

        if action == ModificationType.INSERT:
            sequence, structure = self._insert_stem_pair(
                sequence, structure, node_name, coords, bulge_graph
            )
            self.action_counts.stem_insertions += 1
            self.action_counts.total_insertions += 1
        else:
            sequence, structure = self._delete_stem_pair(
                sequence, structure, node_name, coords, bulge_graph
            )
            self.action_counts.stem_deletions += 1
            self.action_counts.total_deletions += 1

        self.modification_counts[node_name] = (
            self.modification_counts.get(node_name, 0) + 1
        )
        self.type_mod_counts.stem += 1

        return sequence, structure

    def _modify_loops(
        self,
        sequence: str,
        structure: str,
        bulge_graph: BulgeGraph,
        node_type: NodeType,
    ) -> Tuple[str, str]:
        """Apply a single modification to a loop type."""
        if node_type == NodeType.HAIRPIN:
            min_size = self.args.hloop_min_size
            max_size = self.args.hloop_max_size
            max_mods = self.args.same_hloop_max_n_mod
            counter_attr = "hloop"
        elif node_type == NodeType.INTERNAL:
            min_size = self.args.iloop_min_size
            max_size = self.args.iloop_max_size
            max_mods = self.args.same_iloop_max_n_mod
            counter_attr = "iloop"
        elif node_type == NodeType.BULGE:
            min_size = self.args.bulge_min_size
            max_size = self.args.bulge_max_size
            max_mods = self.args.same_bulge_max_n_mod
            counter_attr = "bulge"
        elif node_type == NodeType.MULTI:
            min_size = self.args.mloop_min_size
            max_size = self.args.mloop_max_size
            max_mods = self.args.same_mloop_max_n_mod
            counter_attr = "mloop"
        else:
            return sequence, structure

        eligible = self._get_eligible_nodes(
            bulge_graph, node_type, min_size, max_size, max_mods
        )
        if not eligible:
            return sequence, structure

        node_name, coords = random.choice(eligible)
        action = self._choose_action(
            node_name, coords, node_type, min_size, max_size
        )

        insert_attr = f"{counter_attr}_insertions"
        delete_attr = f"{counter_attr}_deletions"

        if action == ModificationType.INSERT:
            sequence, structure = self._insert_loop_base(
                sequence, structure, node_name, coords, bulge_graph
            )
            setattr(self.action_counts, insert_attr, getattr(self.action_counts, insert_attr) + 1)
            self.action_counts.total_insertions += 1
        else:
            sequence, structure = self._delete_loop_base(
                sequence,
                structure,
                node_name,
                coords,
                node_type,
                min_size,
                bulge_graph,
            )
            setattr(self.action_counts, delete_attr, getattr(self.action_counts, delete_attr) + 1)
            self.action_counts.total_deletions += 1

        self.modification_counts[node_name] = (
            self.modification_counts.get(node_name, 0) + 1
        )

        setattr(self.type_mod_counts, counter_attr, getattr(self.type_mod_counts, counter_attr) + 1)

        return sequence, structure
    
    def _get_nucleotide_size(self, node_type: NodeType, coords: List[int]) -> int:
        """Calculate the nucleotide size of a structure based on its coordinates."""
        if not coords:
            return 0
            
        if node_type == NodeType.STEM:
            # Stems: coords[1] - coords[0] (length of one side)
            if len(coords) >= 2:
                return coords[1] - coords[0] + 1
            return 0
            
        elif node_type == NodeType.HAIRPIN:
            # Hairpin loops: coords[1] - coords[0]
            if len(coords) >= 2:
                return coords[1] - coords[0] + 1
            return 0
            
        elif node_type == NodeType.BULGE:
            # Bulge loops: coords[1] - coords[0]
            if len(coords) >= 2:
                return coords[1] - coords[0] + 1
            return 0
            
        elif node_type == NodeType.MULTI:
            # Multiloops: coords[1] - coords[0] when len(coords) == 2, otherwise size is 0
            if len(coords) == 2:
                return coords[1] - coords[0] + 1
            return 0
            
        elif node_type == NodeType.INTERNAL:
            # Internal loops: we'll need both sides for min/max comparison
            if len(coords) >= 4:
                side1_size = coords[1] - coords[0] + 1
                side2_size = coords[3] - coords[2] + 1
                # Return total size for general purposes, but we'll handle min/max separately
                return side1_size + side2_size
            return 0
            
        return 0

    def _get_eligible_nodes(self, bulge_graph: BulgeGraph, node_type: NodeType,
                           min_size: int, max_size: int, max_mods: int) -> List[Tuple[str, List[int]]]:
        """Get nodes eligible for modification."""
        eligible = []
        nodes_by_type = bulge_graph.get_nodes_by_type(node_type)
        
        for node_name, coords in nodes_by_type:
            # Skip nodes with empty coordinates - they cannot be modified
            if not coords or len(coords) == 0:
                logger.debug(f"Skipping node '{node_name}' with empty coordinates")
                continue
                
            # For multiloops with 0 coordinates, skip them for deletion
            if node_type == NodeType.MULTI and len(coords) == 0:
                logger.debug(f"Skipping multiloop '{node_name}' with 0 coordinates")
                continue
                
            # Check modification count limit
            if self.modification_counts.get(node_name, 0) >= max_mods:
                continue
            
            # Calculate actual nucleotide size
            current_size = self._get_nucleotide_size(node_type, coords)
            
            # Special handling for internal loops
            if node_type == NodeType.INTERNAL and len(coords) >= 4:
                side1_size = coords[1] - coords[0] + 1
                side2_size = coords[3] - coords[2] + 1
                # Check if node can be modified based on min/max constraints
                if self._can_modify_internal_loop(node_name, side1_size, side2_size, min_size, max_size):
                    eligible.append((node_name, coords))
            else:
                # Check if node can be modified
                if self._can_modify_node(node_name, current_size, min_size, max_size):
                    eligible.append((node_name, coords))
        
        return eligible
    
    def _can_modify_internal_loop(self, node_name: str, side1_size: int, side2_size: int, min_size: int, max_size: int) -> bool:
        """Check if an internal loop can be modified based on its two sides."""
        if node_name in self.node_actions:
            # Action already committed
            action = self.node_actions[node_name]
            if action == ModificationType.INSERT:
                # Can insert if max of both sides is less than max_size
                return max(side1_size, side2_size) < max_size
            elif action == ModificationType.DELETE:
                # Can delete if min of both sides is greater than min_size
                return min(side1_size, side2_size) > min_size
        else:
            # Can modify if insertion or deletion is possible
            can_insert = max(side1_size, side2_size) < max_size
            can_delete = min(side1_size, side2_size) > min_size
            return can_insert or can_delete
        
        return False
    
    def _can_modify_node(self, node_name: str, current_size: int, min_size: int, max_size: int) -> bool:
        """Check if a node can be modified."""
        if node_name in self.node_actions:
            # Action already committed
            action = self.node_actions[node_name]
            if action == ModificationType.INSERT:
                return current_size < max_size
            elif action == ModificationType.DELETE:
                return current_size > min_size
        else:
            # Can modify if insertion or deletion is possible
            return current_size < max_size or current_size > min_size
        
        return False
    
    def _choose_action(self, node_name: str, coords: List[int], node_type: NodeType, min_size: int, max_size: int) -> ModificationType:
        """Choose insertion or deletion for a node."""
        if node_name in self.node_actions:
            return self.node_actions[node_name]
        
        # Special handling for internal loops
        if node_type == NodeType.INTERNAL and len(coords) >= 4:
            side1_size = coords[1] - coords[0] + 1
            side2_size = coords[3] - coords[2] + 1
            can_insert = max(side1_size, side2_size) < max_size
            can_delete = min(side1_size, side2_size) > min_size
        else:
            current_size = self._get_nucleotide_size(node_type, coords)
            can_insert = current_size < max_size
            can_delete = current_size > min_size
        
        if can_insert and can_delete:
            action = random.choice([ModificationType.INSERT, ModificationType.DELETE])
        elif can_insert:
            action = ModificationType.INSERT
        elif can_delete:
            action = ModificationType.DELETE
        else:
            # Fallback, though this state implies no modification is possible,
            # which should be caught by _get_eligible_nodes.
            action = ModificationType.INSERT

        self.node_actions[node_name] = action
        return action
    
    def _insert_stem_pair(self, sequence: str, structure: str, node_name: str,
                          coords: List[int], bulge_graph: BulgeGraph) -> Tuple[str, str]:
        """Insert a complementary base pair in a stem at random positions."""
        if len(coords) < 2:
            return sequence, structure
        
        sorted_coords = sorted(coords)
        n = len(sorted_coords)
        
        # Split into left and right halves
        left_half = sorted_coords[:n//2]
        right_half = sorted_coords[n//2:]
        
        # Choose a random position in the left half to insert after
        # We can insert between any adjacent positions in the left half
        left_insert_options = []
        for i in range(len(left_half)):
            if i == 0:
                # Can insert before the first position
                left_insert_options.append(left_half[i] - 1)
            # Can insert after each position
            left_insert_options.append(left_half[i])
        
        # Choose random insertion position in left half
        left_pos = random.choice(left_insert_options)
        
        # Calculate corresponding position in right half
        # If we insert at position left_pos, we need to find the corresponding right position
        # The pairing logic: if left_half[i] pairs with right_half[-(i+1)]
        # We need to find which left position corresponds to our insertion
        
        # Find which "gap" in the left half we're inserting into
        left_gap_index = 0
        for i, pos in enumerate(left_half):
            if left_pos <= pos:
                left_gap_index = i
                break
        else:
            left_gap_index = len(left_half)
        
        # The corresponding right position is the mirror of the left gap
        if left_gap_index == 0:
            # Inserting before first left base, so insert after last right base
            right_pos = right_half[-1]
        elif left_gap_index == len(left_half):
            # Inserting after last left base, so insert before first right base
            right_pos = right_half[0] - 1
        else:
            # Inserting between left positions, insert between corresponding right positions
            # If inserting between left_half[i-1] and left_half[i], 
            # insert between right_half[-(i)] and right_half[-(i+1)]
            right_mirror_index = len(right_half) - left_gap_index
            if right_mirror_index > 0:
                right_pos = right_half[right_mirror_index - 1]
            else:
                right_pos = right_half[0] - 1
        
        # Choose complementary bases
        complement_pairs = [('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G')]
        base_left, base_right = random.choice(complement_pairs)
        
        logger.debug(f"Inserting stem pair at positions {left_pos} ('{base_left}') and {right_pos} ('{base_right}')")
        
        # Insert bases (higher index first to avoid position shifts)
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

        # Update bulge graph indices
        from .bulge_graph_updater import BulgeGraphUpdater
        BulgeGraphUpdater.insert_stem_pair(bulge_graph, node_name, left_pos, right_pos)

        return sequence, structure
    
    def _delete_stem_pair(self, sequence: str, structure: str, node_name: str,
                          coords: List[int], bulge_graph: BulgeGraph) -> Tuple[str, str]:
        """Delete a complementary base pair from a stem at random positions."""
        if len(coords) < 4:  # Need at least 4 bases to safely delete a pair
            return sequence, structure
        
        sorted_coords = sorted(coords)
        n = len(sorted_coords)
        
        # Split into left and right halves
        left_half = sorted_coords[:n//2]
        right_half = sorted_coords[n//2:]
        
        # Choose a random position from the left half to delete
        left_delete_idx = random.randint(0, len(left_half) - 1)
        left_pos = left_half[left_delete_idx] - 1  # Convert to 0-based
        
        # Calculate the corresponding position in the right half
        # The pairing logic: left_half[i] pairs with right_half[-(i+1)]
        # So left_half[left_delete_idx] pairs with right_half[-(left_delete_idx+1)]
        right_delete_idx = len(right_half) - 1 - left_delete_idx
        right_pos = right_half[right_delete_idx] - 1  # Convert to 0-based
        
        logger.debug(f"Deleting stem pair at positions {left_pos} and {right_pos}")
        logger.debug(f"Left half: {left_half}, deleting index {left_delete_idx}")
        logger.debug(f"Right half: {right_half}, deleting index {right_delete_idx}")
        
        # Delete bases (higher index first to avoid position shifts)
        if right_pos < len(sequence) and left_pos < len(sequence) and right_pos > left_pos:
            sequence = sequence[:right_pos] + sequence[right_pos+1:]
            structure = structure[:right_pos] + structure[right_pos+1:]
            sequence = sequence[:left_pos] + sequence[left_pos+1:]
            structure = structure[:left_pos] + structure[left_pos+1:]
        elif left_pos < len(sequence) and right_pos < len(sequence) and left_pos > right_pos:
            # Handle case where positions might be reversed
            sequence = sequence[:left_pos] + sequence[left_pos+1:]
            structure = structure[:left_pos] + structure[left_pos+1:]
            sequence = sequence[:right_pos] + sequence[right_pos+1:]
            structure = structure[:right_pos] + structure[right_pos+1:]

        # Update bulge graph indices
        from .bulge_graph_updater import BulgeGraphUpdater
        BulgeGraphUpdater.delete_stem_pair(bulge_graph, node_name, left_pos, right_pos)

        return sequence, structure
    
    def _insert_loop_base(self, sequence: str, structure: str, node_name: str,
                          coords: List[int], bulge_graph: BulgeGraph) -> Tuple[str, str]:
        """Insert a base in a loop region."""
        if not coords:
            return sequence, structure
        
        # Choose random position within the loop (convert to 0-based)
        pos = random.choice(coords) - 1
        
        # Insert random base
        base = random.choice(['A', 'C', 'G', 'U'])
        sequence = sequence[:pos] + base + sequence[pos:]
        structure = structure[:pos] + '.' + structure[pos:]

        from .bulge_graph_updater import BulgeGraphUpdater
        BulgeGraphUpdater.insert_loop_base(bulge_graph, node_name, pos)

        return sequence, structure
    
    def _delete_loop_base(self, sequence: str, structure: str, node_name: str,
                          coords: List[int], node_type: NodeType, min_size: int, bulge_graph: BulgeGraph) -> Tuple[str, str]:
        """Delete a base from a loop region."""
        if not coords:
            return sequence, structure

        # Calculate actual nucleotide size based on structure type
        if node_type == NodeType.INTERNAL and len(coords) >= 4:
            # For internal loops, check both sides
            side1_size = coords[1] - coords[0] + 1
            side2_size = coords[3] - coords[2] + 1
            # Can delete if the minimum side is greater than min_size
            if min(side1_size, side2_size) <= min_size:
                logger.debug(f"Skipping deletion for internal loop {node_name} as min side size ({min(side1_size, side2_size)}) is not greater than min_size ({min_size})")
                return sequence, structure
        else:
            # For other loop types, calculate size normally
            current_size = self._get_nucleotide_size(node_type, coords)
            if current_size <= min_size:
                logger.debug(f"Skipping deletion for {node_type.value} {node_name} as its size ({current_size}) is not greater than min_size ({min_size})")
                return sequence, structure
        
        # Choose random position to delete (convert to 0-based)
        pos = random.choice(coords) - 1
        
        if pos < len(sequence):
            sequence = sequence[:pos] + sequence[pos+1:]
            structure = structure[:pos] + structure[pos+1:]
            from .bulge_graph_updater import BulgeGraphUpdater
            BulgeGraphUpdater.delete_loop_base(bulge_graph, node_name, pos)

        return sequence, structure
    
    def _calculate_modification_counts(self, bulge_graph: BulgeGraph) -> ModificationCounts:
        """Return modification counts accumulated during modifications."""
        counts = ModificationCounts(
            stem=self.type_mod_counts.stem,
            hloop=self.type_mod_counts.hloop,
            iloop=self.type_mod_counts.iloop,
            bulge=self.type_mod_counts.bulge,
            mloop=self.type_mod_counts.mloop,
        )
        counts.total = (
            counts.stem + counts.hloop + counts.iloop + counts.bulge + counts.mloop
        )
        return counts

    def _calculate_action_counts(self) -> ActionCounts:
        """Return insertion and deletion action counts."""
        return ActionCounts(
            total_insertions=self.action_counts.total_insertions,
            total_deletions=self.action_counts.total_deletions,
            stem_insertions=self.action_counts.stem_insertions,
            stem_deletions=self.action_counts.stem_deletions,
            hloop_insertions=self.action_counts.hloop_insertions,
            hloop_deletions=self.action_counts.hloop_deletions,
            iloop_insertions=self.action_counts.iloop_insertions,
            iloop_deletions=self.action_counts.iloop_deletions,
            bulge_insertions=self.action_counts.bulge_insertions,
            bulge_deletions=self.action_counts.bulge_deletions,
            mloop_insertions=self.action_counts.mloop_insertions,
            mloop_deletions=self.action_counts.mloop_deletions,
        )
