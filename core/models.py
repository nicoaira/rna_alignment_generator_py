"""
Data models for the RNA triplet generator.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import uuid
from datetime import datetime


class NodeType(Enum):
    """Types of RNA structural nodes."""
    STEM = "stem"
    HAIRPIN = "hairpin"
    INTERNAL = "internal"
    MULTI = "multi"
    BULGE = "bulge"
    FIVE_PRIME = "five_prime"
    THREE_PRIME = "three_prime"
    UNKNOWN = "unknown"


class ModificationType(Enum):
    """Types of modifications that can be applied."""
    INSERT = "insert"
    DELETE = "delete"
    INSERT_PAIR = "insert_pair"
    DELETE_PAIR = "delete_pair"


@dataclass
class GraphNode:
    """Represents a node in the RNA bulge graph."""
    positions: List[int]
    start: Optional[int] = None
    end: Optional[int] = None
    
    def __len__(self) -> int:
        """Return the size of the node (number of positions)."""
        return len(self.positions)


@dataclass
class BulgeGraph:
    """Represents the bulge graph structure of an RNA molecule."""
    elements: Dict[str, GraphNode]
    
    def node_mapping(self) -> Dict[str, List[int]]:
        """Get the node mapping (node name -> positions)."""
        return {name: node.positions for name, node in self.elements.items()}
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[Tuple[str, List[int]]]:
        """Get all nodes of a specific type."""
        result = []
        node_mapping = self.node_mapping()
        
        for node_name, coords in node_mapping.items():
            if classify_node(node_name, coords) == node_type:
                result.append((node_name, coords))
        
        return result


@dataclass
class ModificationCounts:
    """Tracks the number of modifications applied to each structural element."""
    total: int = 0
    stem: int = 0
    hloop: int = 0
    iloop: int = 0
    bulge: int = 0
    mloop: int = 0


@dataclass
class SampledModifications:
    """Holds sampled modification counts for generating a positive sample."""
    n_stem_indels: int
    n_hloop_indels: int
    n_iloop_indels: int
    n_bulge_indels: int
    n_mloop_indels: int


@dataclass
class ActionCounts:
    """Tracks counts of insertion and deletion actions by structure type."""
    total_insertions: int = 0
    total_deletions: int = 0
    stem_insertions: int = 0
    stem_deletions: int = 0
    hloop_insertions: int = 0
    hloop_deletions: int = 0
    iloop_insertions: int = 0
    iloop_deletions: int = 0
    bulge_insertions: int = 0
    bulge_deletions: int = 0
    mloop_insertions: int = 0
    mloop_deletions: int = 0


@dataclass
class RnaTriplet:
    """Represents a complete RNA triplet with anchor, positive, and negative samples."""
    triplet_id: int
    anchor_seq: str
    anchor_structure: str
    positive_seq: str
    positive_structure: str
    negative_seq: str
    negative_structure: str
    total_modifications: int
    stem_modifications: int
    hloop_modifications: int
    iloop_modifications: int
    bulge_modifications: int
    mloop_modifications: int
    total_len_stem: int
    total_len_hloop: int
    total_len_iloop: int
    total_len_bulge: int
    total_len_mloop: int
    anchor_seq_len: int
    positive_seq_len: int
    negative_seq_len: int
    stem_insertions: int
    stem_deletions: int
    hloop_insertions: int
    hloop_deletions: int
    iloop_insertions: int
    iloop_deletions: int
    bulge_insertions: int
    bulge_deletions: int
    mloop_insertions: int
    mloop_deletions: int
    total_insertions: int
    total_deletions: int
    f_stem_modifications: float
    f_stem_insertions: float
    f_stem_deletions: float
    f_hloop_modifications: float
    f_hloop_insertions: float
    f_hloop_deletions: float
    f_iloop_modifications: float
    f_iloop_insertions: float
    f_iloop_deletions: float
    f_bulge_modifications: float
    f_bulge_insertions: float
    f_bulge_deletions: float
    f_mloop_modifications: float
    f_mloop_insertions: float
    f_mloop_deletions: float
    f_total_modifications: float
    f_total_insertions: float
    f_total_deletions: float


@dataclass
class DatasetMetadata:
    """Metadata for the generated dataset."""
    run_id: str
    timestamp: str
    parameters: Dict
    num_triplets: int
    
    @classmethod
    def create(cls, args, num_triplets: int) -> 'DatasetMetadata':
        """Create metadata from command-line arguments."""
        return cls(
            run_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            parameters=vars(args),
            num_triplets=num_triplets
        )


def classify_node(node_name: str, coords: List[int]) -> NodeType:
    """
    Classify a node by its name and coordinate count.
    
    Args:
        node_name: Name of the node (e.g., 's0', 'h1', 'i2')
        coords: List of coordinate positions
        
    Returns:
        NodeType classification
    """
    if node_name.startswith('s'):
        return NodeType.STEM
    elif node_name.startswith('h'):
        return NodeType.HAIRPIN
    elif node_name.startswith('i'):
        # Internal loops have 4 coordinates, bulges have 2
        if len(coords) == 4:
            return NodeType.INTERNAL
        elif len(coords) == 2:
            return NodeType.BULGE
        else:
            return NodeType.INTERNAL
    elif node_name.startswith('m'):
        return NodeType.MULTI
    elif node_name == "f0":
        return NodeType.FIVE_PRIME
    elif node_name == "t0":
        return NodeType.THREE_PRIME
    else:
        # For other nodes, differentiate by coordinate count
        if len(coords) == 4:
            return NodeType.INTERNAL
        elif len(coords) == 2:
            return NodeType.BULGE
        else:
            return NodeType.INTERNAL