#!/usr/bin/env python3
"""
create_test_graphs.py

Generates test cases for bulge graph updates by creating sequences,
applying single modifications, and outputting before/after states in JSON format.
This script creates graphs with forgi for comparison with our implementation.
"""

import argparse
import json
import logging
import random
from typing import Dict, List, Any
from pathlib import Path

from core.rna_generator import RnaGenerator, BulgeGraphParser
from core.modification_engine import ModificationEngine
from core.models import NodeType, ModificationType, classify_node
from utils.logger import setup_logging

import forgi.graph.bulge_graph as fgb
FORGI_AVAILABLE = True


def create_mock_args():
    """Create a mock args object with default modification parameters."""
    class MockArgs:
        def __init__(self):
            # Stem parameters
            self.stem_min_size = 2
            self.stem_max_size = 20
            self.same_stem_max_n_mod = 1
            
            # Hairpin loop parameters
            self.hloop_min_size = 3
            self.hloop_max_size = 15
            self.same_hloop_max_n_mod = 1
            
            # Internal loop parameters
            self.iloop_min_size = 2
            self.iloop_max_size = 15
            self.same_iloop_max_n_mod = 1
            
            # Bulge parameters
            self.bulge_min_size = 1
            self.bulge_max_size = 10
            self.same_bulge_max_n_mod = 1
            
            # Multi loop parameters
            self.mloop_min_size = 2
            self.mloop_max_size = 20
            self.same_mloop_max_n_mod = 1
            
            # Other parameters
            self.mod_normalization = False
            self.normalization_len = 100
    
    return MockArgs()


def forgi_bulge_graph_to_dict(forgi_bg) -> Dict[str, List[int]]:
    """Convert a forgi BulgeGraph to a dictionary format."""
    if not FORGI_AVAILABLE or not forgi_bg:
        return {}
    
    result = {}
    for elem, positions in forgi_bg.defines.items():
        result[elem] = sorted(positions) if positions else []
    
    return result


def bulge_graph_to_dict(bulge_graph) -> Dict[str, List[int]]:
    """Convert a BulgeGraph to a dictionary format."""
    return bulge_graph.node_mapping()


def create_forgi_bulge_graph(structure: str, sequence: str = None) -> Dict[str, List[int]]:
    """Create a forgi bulge graph directly and return as dict."""
    if not FORGI_AVAILABLE:
        return {}
    
    try:
        forgi_bg = (fgb.BulgeGraph.from_dotbracket(structure, sequence)
                   if sequence else
                   fgb.BulgeGraph.from_dotbracket(structure))
        return forgi_bulge_graph_to_dict(forgi_bg)
    except Exception as e:
        logging.debug(f"Failed to create forgi bulge graph: {e}")
        return {}


def would_deletion_cause_disappearance(node_type: NodeType, coords: List[int], action: ModificationType) -> bool:
    """
    Check if a deletion operation would cause a structural element to disappear.
    
    Args:
        node_type: Type of the node (STEM, HAIRPIN, etc.)
        coords: Coordinates of the node
        action: The modification action
        
    Returns:
        True if deletion would cause disappearance, False otherwise
    """
    if action != ModificationType.DELETE:
        return False
    
    # Skip deletion of stems with length 1 to avoid forgi renaming issues
    if (node_type == NodeType.STEM and len(coords) == 4 and 
        coords[0] == coords[1] and coords[2] == coords[3]):
        return True
    
    # Skip deletion of hairpins, multiloops, or bulges with only 1 base
    if node_type in [NodeType.HAIRPIN, NodeType.MULTI, NodeType.BULGE]:
        if len(coords) == 2 and (coords[1] - coords[0] + 1) <= 1:
            return True
    
    # Skip deletion of internal loops where either side has length 1
    if node_type == NodeType.INTERNAL:
        if len(coords) == 4:
            left_side_length = coords[1] - coords[0] + 1
            right_side_length = coords[3] - coords[2] + 1
            if left_side_length == 1 or right_side_length == 1:
                return True
        # Also handle cases where coords might be in different format
        elif len(coords) >= 2:
            # Check for any single-position segments
            unique_coords = sorted(set(coords))
            for i in range(0, len(unique_coords), 2):
                if i + 1 < len(unique_coords) and unique_coords[i] == unique_coords[i + 1]:
                    return True
    
    return False


def apply_single_modification(sequence: str, structure: str, modification_engine: ModificationEngine, 
                            bulge_graph, target_node_type: NodeType = None) -> Dict[str, Any]:
    """
    Apply a single modification to a sequence and return the results.
    
    Args:
        sequence: Original RNA sequence
        structure: Original dot-bracket structure
        modification_engine: ModificationEngine instance
        bulge_graph: Original bulge graph
        target_node_type: Specific node type to target (optional)
        
    Returns:
        Dictionary with modification results or None if no modification applied
    """
    # Get eligible nodes for modification
    all_node_types = [NodeType.STEM, NodeType.HAIRPIN, NodeType.INTERNAL, NodeType.BULGE, NodeType.MULTI]
    
    if target_node_type:
        node_types_to_try = [target_node_type]
    else:
        # Randomize the order of node types to ensure balanced distribution
        node_types_to_try = all_node_types.copy()
        random.shuffle(node_types_to_try)
    
    # Collect all eligible nodes from all types first
    all_eligible_nodes = []
    
    for node_type in node_types_to_try:
        # Get size constraints based on node type
        if node_type == NodeType.STEM:
            min_size, max_size, max_mods = (modification_engine.args.stem_min_size, 
                                          modification_engine.args.stem_max_size,
                                          modification_engine.args.same_stem_max_n_mod)
        elif node_type == NodeType.HAIRPIN:
            min_size, max_size, max_mods = (modification_engine.args.hloop_min_size,
                                          modification_engine.args.hloop_max_size,
                                          modification_engine.args.same_hloop_max_n_mod)
        elif node_type == NodeType.INTERNAL:
            min_size, max_size, max_mods = (modification_engine.args.iloop_min_size,
                                          modification_engine.args.iloop_max_size,
                                          modification_engine.args.same_iloop_max_n_mod)
        elif node_type == NodeType.BULGE:
            min_size, max_size, max_mods = (modification_engine.args.bulge_min_size,
                                          modification_engine.args.bulge_max_size,
                                          modification_engine.args.same_bulge_max_n_mod)
        elif node_type == NodeType.MULTI:
            min_size, max_size, max_mods = (modification_engine.args.mloop_min_size,
                                          modification_engine.args.mloop_max_size,
                                          modification_engine.args.same_mloop_max_n_mod)
        else:
            continue
        
        # Get eligible nodes of this type
        eligible_nodes = modification_engine._get_eligible_nodes(bulge_graph, node_type, 
                                                               min_size, max_size, max_mods)
        
        # Add node type info to each eligible node
        for node_name, coords in eligible_nodes:
            all_eligible_nodes.append((node_type, node_name, coords, min_size, max_size))
    
    if not all_eligible_nodes:
        return None
    
    # Try to find a valid modification, filtering out problematic deletions
    max_attempts = len(all_eligible_nodes) * 2  # Allow multiple attempts
    
    for attempt in range(max_attempts):
        # Randomly select from all eligible nodes across all types
        node_type, node_name, coords, min_size, max_size = random.choice(all_eligible_nodes)
        
        # Reset the node_actions to ensure no bias from previous test cases
        modification_engine.node_actions = {}
        
        # Choose action (insert or delete)
        action = modification_engine._choose_action(node_name, coords, node_type, min_size, max_size)
        
        # Check if this deletion would cause structural disappearance
        if would_deletion_cause_disappearance(node_type, coords, action):
            logging.debug(f"Skipping deletion of {node_type.value} {node_name} with coords {coords} - would cause disappearance")
            continue
        
        # Apply the modification with correct parameters
        try:
            original_coords_full = coords[:]  # Capture original coordinates
            if node_type == NodeType.STEM:
                if action == ModificationType.INSERT:
                    new_seq, new_struct = modification_engine._insert_stem_pair(sequence, structure, node_name, coords, bulge_graph)
                    action_name = "insert_pair"
                else:
                    new_seq, new_struct = modification_engine._delete_stem_pair(sequence, structure, node_name, coords, bulge_graph)
                    action_name = "delete_pair"
            else:
                if action == ModificationType.INSERT:
                    new_seq, new_struct = modification_engine._insert_loop_base(sequence, structure, node_name, coords, bulge_graph)
                    action_name = "insert"
                else:
                    new_seq, new_struct = modification_engine._delete_loop_base(sequence, structure, node_name, coords, node_type, min_size, bulge_graph)
                    action_name = "delete"
            
            # Check if modification actually occurred
            if new_seq != sequence or new_struct != structure:
                return {
                    'modified_node': node_name,
                    'node_type': node_type.value,
                    'action': action_name,
                    'new_sequence': new_seq,
                    'new_structure': new_struct,
                    'original_coords': original_coords_full,
                }
        except Exception as e:
            logging.debug(f"Failed to apply {action_name} to {node_type.value} {node_name}: {e}")
            continue
    
    logging.debug(f"Could not find valid modification after {max_attempts} attempts")
    return None


def generate_test_case(seq_min_len: int, seq_max_len: int, seq_len_mean: float, 
                      seq_len_sd: float, seq_len_distribution: str, 
                      modification_engine: ModificationEngine, bulge_parser: BulgeGraphParser,
                      case_id: int) -> Dict[str, Any]:
    """Generate a single test case."""
    max_attempts = 50
    
    for attempt in range(max_attempts):
        try:
            # Generate sequence
            if seq_len_distribution == "norm":
                length = RnaGenerator.choose_sequence_length(seq_len_distribution, seq_min_len, seq_max_len, seq_len_mean, seq_len_sd)
            else:
                length = random.randint(seq_min_len, seq_max_len)
            
            sequence = RnaGenerator.generate_random_sequence(length)
            
            # Fold sequence
            structure = RnaGenerator.fold_rna(sequence)
            if not structure:
                continue
            
            # Create forgi bulge graphs (reference implementation)
            pre_mod_forgi_bg = create_forgi_bulge_graph(structure, sequence)
            if not pre_mod_forgi_bg and FORGI_AVAILABLE:
                continue
            
            # Parse bulge graph with our implementation for modification purposes
            bulge_graph = bulge_parser.parse_structure(structure, sequence)
            if not bulge_graph or not bulge_graph.elements:
                continue
            
            # Apply single modification
            mod_result = apply_single_modification(sequence, structure, modification_engine, bulge_graph)
            if not mod_result:
                continue
            
            # Create forgi bulge graph for the modified structure
            post_mod_forgi_bg = create_forgi_bulge_graph(mod_result['new_structure'], mod_result['new_sequence'])
            if not post_mod_forgi_bg and FORGI_AVAILABLE:
                continue
            
            # Create test case with forgi as the reference (expected format for test_bulge_graph_updater.py)
            test_case = {
                'case_id': case_id,
                'pre_mod_seq': sequence,
                'pre_mod_dotbracket': structure,
                # Use forgi as the reference implementation for the test
                'pre_mod_bulgegraph': pre_mod_forgi_bg,
                'post_mod_bulgegraph': post_mod_forgi_bg,
                # Modification details
                'mod_node': mod_result['modified_node'],
                'mod_node_type': mod_result['node_type'],
                'mod_action': mod_result['action'],
                'original_coords': mod_result['original_coords'],
                'post_mod_seq': mod_result['new_sequence'],
                'post_mod_dotbracket': mod_result['new_structure'],
                'sequence_length_change': len(mod_result['new_sequence']) - len(sequence),
                'attempt': attempt + 1,
                'forgi_available': FORGI_AVAILABLE
            }
            
            return test_case
            
        except Exception as e:
            logging.debug(f"Attempt {attempt + 1} failed for case {case_id}: {e}")
            continue
    
    logging.warning(f"Failed to generate test case {case_id} after {max_attempts} attempts")
    return None


def main():
    """Main function to generate test cases."""
    parser = argparse.ArgumentParser(
        description="Generate test cases for bulge graph updates"
    )
    
    parser.add_argument("--n", type=int, default=100,
                       help="Number of test cases to generate")
    parser.add_argument("--seq_min_len", type=int, default=50,
                       help="Minimum sequence length")
    parser.add_argument("--seq_max_len", type=int, default=100,
                       help="Maximum sequence length")
    parser.add_argument("--seq_len_mean", type=float, default=75.0,
                       help="Mean for normal distribution of sequence length")
    parser.add_argument("--seq_len_sd", type=float, default=10.0,
                       help="Standard deviation for normal distribution of sequence length")
    parser.add_argument("--seq_len_distribution", choices=["unif", "norm"], default="unif",
                       help="Distribution of sequence lengths")
    parser.add_argument("--output", type=str, default="test_cases.json",
                       help="Output JSON file")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    print("Starting test case generation...")
    
    # Set up logging
    setup_logging(debug=args.debug)
    logger = logging.getLogger(__name__)
    
    print("Logging set up...")
    
    logger.info(f"Generating {args.n} test cases for bulge graph updates")
    logger.info(f"Sequence length: {args.seq_min_len}-{args.seq_max_len} ({args.seq_len_distribution} distribution)")
    
    print("Initializing components...")
    
    # Initialize components
    mock_args = create_mock_args()
    print("Mock args created...")
    
    bulge_parser = BulgeGraphParser()
    print("Bulge parser created...")
    
    modification_engine = ModificationEngine(mock_args)
    print("Modification engine created...")
    
    # Generate test cases
    test_cases = []
    successful_cases = 0
    
    for i in range(args.n):
        logger.debug(f"Generating test case {i + 1}/{args.n}")
        
        test_case = generate_test_case(
            args.seq_min_len, args.seq_max_len, args.seq_len_mean, 
            args.seq_len_sd, args.seq_len_distribution, 
            modification_engine, bulge_parser, i + 1
        )
        
        if test_case:
            test_cases.append(test_case)
            successful_cases += 1
            logger.debug(f"Generated case {i + 1}: {test_case['mod_node']} ({test_case['mod_action']})")
        
        # Progress update
        if (i + 1) % 10 == 0:
            logger.info(f"Progress: {i + 1}/{args.n} attempts, {successful_cases} successful")
    
    # Create output data
    output_data = {
        'metadata': {
            'total_cases': len(test_cases),
            'successful_rate': len(test_cases) / args.n,
            'parameters': {
                'n': args.n,
                'seq_min_len': args.seq_min_len,
                'seq_max_len': args.seq_max_len,
                'seq_len_mean': args.seq_len_mean,
                'seq_len_sd': args.seq_len_sd,
                'seq_len_distribution': args.seq_len_distribution
            }
        },
        'test_cases': test_cases
    }
    
    # Add statistics
    if test_cases:
        mod_types = {}
        mod_actions = {}
        length_changes = []
        
        for case in test_cases:
            mod_types[case['mod_node_type']] = mod_types.get(case['mod_node_type'], 0) + 1
            mod_actions[case['mod_action']] = mod_actions.get(case['mod_action'], 0) + 1
            length_changes.append(case['sequence_length_change'])
        
        output_data['metadata']['statistics'] = {
            'modification_types': mod_types,
            'modification_actions': mod_actions,
            'length_changes': {
                'min': min(length_changes),
                'max': max(length_changes),
                'mean': sum(length_changes) / len(length_changes)
            }
        }
    
    # Save to file
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Generated {len(test_cases)} test cases")
    logger.info(f"Success rate: {len(test_cases)}/{args.n} ({len(test_cases)/args.n*100:.1f}%)")
    logger.info(f"Output saved to: {output_path}")
    
    if test_cases:
        logger.info("Statistics:")
        stats = output_data['metadata']['statistics']
        logger.info(f"  Modification types: {stats['modification_types']}")
        logger.info(f"  Modification actions: {stats['modification_actions']}")
        logger.info(f"  Length changes: {stats['length_changes']}")

if __name__ == "__main__":
    main()