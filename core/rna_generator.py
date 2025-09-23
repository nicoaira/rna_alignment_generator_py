"""
RNA sequence generation and folding utilities.
"""

import logging
import random
import subprocess
from typing import Dict, List, Tuple
import numpy as np
from scipy.stats import truncnorm

# Import forgi directly instead of using subprocess
try:
    import forgi.graph.bulge_graph as fgb
    FORGI_AVAILABLE = True
except ImportError:
    FORGI_AVAILABLE = False
    logging.warning("forgi not available - bulge graph parsing will be limited")

from .models import BulgeGraph, GraphNode

logger = logging.getLogger(__name__)


class RnaGenerator:
    """Handles RNA sequence generation and folding operations."""
    
    @staticmethod
    def generate_random_sequence(length: int) -> str:
        """
        Generate a random RNA sequence of the given length.
        
        Args:
            length: Length of the sequence to generate
            
        Returns:
            Random RNA sequence with bases A, C, G, U
        """
        bases = ['A', 'C', 'G', 'U']
        return ''.join(random.choice(bases) for _ in range(length))
    
    @staticmethod
    def choose_sequence_length(distribution: str, min_len: int, max_len: int, 
                             mean: float, sd: float) -> int:
        """
        Choose a sequence length from either uniform or normal distribution.
        
        Args:
            distribution: Either 'unif' or 'norm'
            min_len: Minimum length
            max_len: Maximum length
            mean: Mean for normal distribution
            sd: Standard deviation for normal distribution
            
        Returns:
            Selected sequence length
        """
        if distribution == 'unif':
            return random.randint(min_len, max_len)
        elif distribution == 'norm':
            # Use truncated normal distribution
            a, b = (min_len - mean) / sd, (max_len - mean) / sd
            return int(truncnorm.rvs(a, b, loc=mean, scale=sd))
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
    
    @staticmethod
    def fold_rna(
        sequence: str,
        constraints_pairs: List[Tuple[int, int]] | None = None,
        constraints_unpaired: List[int] | None = None,
    ) -> str:
        """
        Fold an RNA sequence using RNAfold to get the MFE structure.
        
        Args:
            sequence: RNA sequence to fold
            constraints_pairs: Optional list of 0-based index pairs to force pairing
            
        Returns:
            Dot-bracket structure string
            
        Raises:
            RuntimeError: If RNAfold fails
        """
        try:
            cmd = ['RNAfold', '--noPS']
            input_text = sequence + '\n'
            constraint_line = None
            has_constraints = bool(constraints_pairs) or bool(constraints_unpaired)
            if has_constraints:
                cons = ['.' for _ in range(len(sequence))]
                if constraints_pairs:
                    for i, j in constraints_pairs:
                        if 0 <= i < len(sequence) and 0 <= j < len(sequence):
                            a, b = (i, j) if i < j else (j, i)
                            cons[a] = '('
                            cons[b] = ')'
                if constraints_unpaired:
                    for idx in constraints_unpaired:
                        if 0 <= idx < len(sequence) and cons[idx] == '.':
                            cons[idx] = 'x'
                constraint_line = ''.join(cons)
                cmd = ['RNAfold', '--noPS', '-C']
                input_text = sequence + '\n' + constraint_line + '\n'

            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(input=input_text)
            
            if process.returncode != 0:
                raise RuntimeError(f"RNAfold failed: {stderr}")
            
            # Parse output - second line contains structure
            lines = stdout.strip().split('\n')
            if len(lines) < 2:
                raise RuntimeError(f"Unexpected RNAfold output: {stdout}")
            
            # Extract structure (everything before the first space)
            structure_line = lines[1].strip()
            structure = structure_line.split()[0]
            
            return structure
            
        except FileNotFoundError:
            raise RuntimeError("RNAfold not found. Please install ViennaRNA package.")
        except Exception as e:
            logger.error(f"Error folding RNA sequence: {e}")
            # Return placeholder structure as fallback
            return '.' * len(sequence)
    
    @staticmethod
    def fold_rna_batch(sequences: List[Tuple[str, str]]) -> Dict[str, str]:
        """
        Fold multiple RNA sequences in a single RNAfold call.
        
        Args:
            sequences: List of (id, sequence) tuples
            
        Returns:
            Dictionary mapping sequence IDs to their structures
        """
        if not sequences:
            return {}
        
        try:
            # Create FASTA input
            fasta_input = []
            for seq_id, sequence in sequences:
                fasta_input.append(f">{seq_id}")
                fasta_input.append(sequence)
            
            input_text = '\n'.join(fasta_input)
            
            # Run RNAfold
            process = subprocess.Popen(
                ['RNAfold', '--noPS'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(input=input_text)
            
            if process.returncode != 0:
                raise RuntimeError(f"RNAfold batch failed: {stderr}")
            
            # Parse output
            results = {}
            lines = stdout.strip().split('\n')
            i = 0
            
            while i < len(lines):
                if lines[i].startswith('>'):
                    seq_id = lines[i][1:].strip()
                    i += 1  # Skip sequence line
                    i += 1  # Get structure line
                    if i < len(lines):
                        structure = lines[i].split()[0]
                        results[seq_id] = structure
                i += 1
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch folding: {e}")
            # Return placeholder structures
            return {seq_id: '.' * len(seq) for seq_id, seq in sequences}


class BulgeGraphParser:
    """Handles parsing of dot-bracket structures into bulge graphs using forgi directly."""
    
    def __init__(self):
        """Initialize the parser."""
        if not FORGI_AVAILABLE:
            logger.warning("forgi not available - using fallback parsing")
    
    def _build_forgi_bulge_graph(self, structure: str, sequence: str = None):
        """Create a forgi BulgeGraph from dot-bracket structure."""
        if not FORGI_AVAILABLE:
            raise ImportError("forgi library not available")
        
        return (fgb.BulgeGraph.from_dotbracket(structure, sequence)
                if sequence else
                fgb.BulgeGraph.from_dotbracket(structure))
    
    def _forgi_to_our_format(self, forgi_bg) -> BulgeGraph:
        """Convert forgi BulgeGraph to our internal format."""
        elements = {}
        
        # Convert forgi elements to our format
        for elem, positions in forgi_bg.defines.items():
            pos_list = sorted(positions) if positions else []
            start = pos_list[0] if pos_list else None
            end = pos_list[-1] if pos_list else None
            
            elements[elem] = GraphNode(
                positions=pos_list,
                start=start,
                end=end
            )
        
        return BulgeGraph(elements=elements)
    
    def parse_structure(self, structure: str, sequence: str = None) -> BulgeGraph:
        """
        Parse a dot-bracket structure into a bulge graph.
        
        Args:
            structure: Dot-bracket structure string
            sequence: Optional RNA sequence (for validation)
            
        Returns:
            BulgeGraph object
        """
        if not structure:
            return BulgeGraph(elements={})
        
        try:
            if FORGI_AVAILABLE:
                # Use forgi directly - much more efficient
                forgi_bg = self._build_forgi_bulge_graph(structure, sequence)
                return self._forgi_to_our_format(forgi_bg)
            else:
                # Fallback to simple parsing if forgi not available
                logger.warning("Using fallback structure parsing - install forgi for better performance")
                return self._fallback_parse(structure)
                
        except Exception as e:
            logger.error(f"Error parsing structure {structure}: {e}")
            return BulgeGraph(elements={})
    
    def _fallback_parse(self, structure: str) -> BulgeGraph:
        """
        Fallback parsing method when forgi is not available.
        This is a simplified parser that only identifies basic stem/loop regions.
        """
        logger.warning("Using simplified fallback parser - install forgi for full functionality")
        
        # This is a very basic implementation
        # In practice, you'd want to encourage users to install forgi
        elements = {}
        
        # Simple stem detection
        stack = []
        pairs = {}
        
        for i, char in enumerate(structure):
            if char == '(':
                stack.append(i + 1)  # 1-based indexing
            elif char == ')':
                if stack:
                    start = stack.pop()
                    pairs[start] = i + 1
        
        # Create simple stem nodes
        if pairs:
            stem_positions = []
            for start, end in pairs.items():
                stem_positions.extend([start, end])
            
            if stem_positions:
                elements['s0'] = GraphNode(
                    positions=sorted(stem_positions),
                    start=min(stem_positions),
                    end=max(stem_positions)
                )
        
        return BulgeGraph(elements=elements)
    
    def parse_structures_batch(self, structures: List[Tuple[str, str]]) -> Dict[str, BulgeGraph]:
        """
        Parse multiple structures efficiently.
        
        Args:
            structures: List of (id, structure) tuples
            
        Returns:
            Dictionary mapping IDs to BulgeGraph objects
        """
        results = {}
        
        for struct_id, structure in structures:
            results[struct_id] = self.parse_structure(structure)
        
        return results
