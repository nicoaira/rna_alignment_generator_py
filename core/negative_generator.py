"""
Negative sample generation for RNA triplets.
"""

import logging
import random
from typing import Tuple
import numpy as np
from scipy.stats import norm

from .rna_generator import RnaGenerator

logger = logging.getLogger(__name__)


class NegativeSampleGenerator:
    """Handles generation of negative samples using dinucleotide shuffling."""
    
    def __init__(self, args):
        """Initialize the negative sample generator."""
        self.args = args
        self.rna_generator = RnaGenerator()
    
    def generate_negative_sample(self, anchor_sequence: str) -> Tuple[str, str]:
        """
        Generate a negative sample from an anchor sequence.
        
        Args:
            anchor_sequence: The original anchor sequence
            
        Returns:
            Tuple of (negative_sequence, negative_structure)
        """
        # Apply dinucleotide shuffling
        neg_seq = self._dinucleotide_shuffle(anchor_sequence)
        
        # Apply length variation if specified
        if self.args.neg_len_variation > 0:
            neg_seq = self._apply_length_variation(neg_seq)
        
        # Fold the negative sequence
        neg_struct = self.rna_generator.fold_rna(neg_seq)
        
        return neg_seq, neg_struct
    
    def _dinucleotide_shuffle(self, sequence: str) -> str:
        """
        Shuffle the sequence while preserving dinucleotide frequencies.
        
        Args:
            sequence: Input RNA sequence
            
        Returns:
            Shuffled sequence with preserved dinucleotide frequencies
        """
        if len(sequence) <= 1:
            return sequence
        
        # Build adjacency graph of dinucleotides
        graph = {}
        chars = list(sequence)
        
        for i in range(len(chars) - 1):
            src = chars[i]
            dst = chars[i + 1]
            
            if src not in graph:
                graph[src] = []
            graph[src].append(dst)
        
        # Shuffle each adjacency list
        for targets in graph.values():
            random.shuffle(targets)
        
        # Construct Eulerian path
        stack = [chars[0]]
        trail = []
        
        while stack:
            current = stack[-1]
            
            if current in graph and graph[current]:
                next_node = graph[current].pop()
                stack.append(next_node)
            else:
                trail.append(stack.pop())
        
        trail.reverse()
        
        # Validate length
        if len(trail) != len(sequence):
            logger.warning("Dinucleotide shuffle failed, returning original sequence")
            return sequence
        
        return ''.join(trail)
    
    def _apply_length_variation(self, sequence: str) -> str:
        """
        Apply random length variation to the sequence.
        
        Args:
            sequence: Input sequence
            
        Returns:
            Sequence with modified length
        """
        variation = random.randint(-self.args.neg_len_variation, self.args.neg_len_variation)
        
        if variation > 0:
            # Lengthen sequence
            for _ in range(variation):
                pos = random.randint(0, len(sequence))
                base = random.choice(['A', 'C', 'G', 'U'])
                sequence = sequence[:pos] + base + sequence[pos:]
        
        elif variation < 0:
            # Shorten sequence
            abs_variation = abs(variation)
            if abs_variation < len(sequence):
                for _ in range(abs_variation):
                    if sequence:
                        pos = random.randint(0, len(sequence) - 1)
                        sequence = sequence[:pos] + sequence[pos + 1:]
        
        return sequence


class AppendingEngine:
    """Handles appending events for RNA sequences."""
    
    def __init__(self, args):
        """Initialize the appending engine."""
        self.args = args
        self.rna_generator = RnaGenerator()
    
    def maybe_append_sequences(self, positive_seq: str, positive_struct: str,
                              negative_seq: str, negative_struct: str,
                              anchor_length: int) -> Tuple[str, str, str, str]:
        """
        Maybe apply appending events to both positive and negative sequences.
        
        Args:
            positive_seq: Positive sequence
            positive_struct: Positive structure
            negative_seq: Negative sequence  
            negative_struct: Negative structure
            anchor_length: Length of the original anchor
            
        Returns:
            Tuple of (modified_pos_seq, modified_pos_struct, modified_neg_seq, modified_neg_struct)
        """
        if random.random() > self.args.appending_event_probability:
            return positive_seq, positive_struct, negative_seq, negative_struct
        
        # Determine appending side
        r = random.random()
        p_both = self.args.both_sides_appending_probability
        p_left = (1.0 - p_both) / 2.0
        p_right = p_left
        
        # Sample append lengths
        mean_append = anchor_length * self.args.appending_size_factor
        sigma_append = max(1.0, mean_append / 2.0)
        
        # Generate linker
        linker_len = random.randint(self.args.linker_min, self.args.linker_max)
        linker_seq = self.rna_generator.generate_random_sequence(linker_len)
        linker_struct = '.' * linker_len
        
        if r < p_left:
            # Append to left side only
            append_len = self._sample_append_length(mean_append, sigma_append)
            append_seq = self.rna_generator.generate_random_sequence(append_len)
            append_struct = self.rna_generator.fold_rna(append_seq)
            
            positive_seq = append_seq + linker_seq + positive_seq
            positive_struct = append_struct + linker_struct + positive_struct
            negative_seq = append_seq + linker_seq + negative_seq
            negative_struct = append_struct + linker_struct + negative_struct
            
        elif r < p_left + p_right:
            # Append to right side only
            append_len = self._sample_append_length(mean_append, sigma_append)
            append_seq = self.rna_generator.generate_random_sequence(append_len)
            append_struct = self.rna_generator.fold_rna(append_seq)
            
            positive_seq = positive_seq + linker_seq + append_seq
            positive_struct = positive_struct + linker_struct + append_struct
            negative_seq = negative_seq + linker_seq + append_seq
            negative_struct = negative_struct + linker_struct + append_struct
            
        else:
            # Append to both sides
            left_len = self._sample_append_length(mean_append, sigma_append)
            left_seq = self.rna_generator.generate_random_sequence(left_len)
            left_struct = self.rna_generator.fold_rna(left_seq)
            
            right_len = self._sample_append_length(mean_append, sigma_append)
            right_seq = self.rna_generator.generate_random_sequence(right_len)
            right_struct = self.rna_generator.fold_rna(right_seq)
            
            positive_seq = left_seq + linker_seq + positive_seq + linker_seq + right_seq
            positive_struct = left_struct + linker_struct + positive_struct + linker_struct + right_struct
            negative_seq = left_seq + linker_seq + negative_seq + linker_seq + right_seq
            negative_struct = left_struct + linker_struct + negative_struct + linker_struct + right_struct
        
        return positive_seq, positive_struct, negative_seq, negative_struct
    
    def _sample_append_length(self, mean: float, sigma: float) -> int:
        """Sample append length from normal distribution."""
        sample = np.random.normal(mean, sigma)
        return max(1, int(round(sample)))