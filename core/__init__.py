"""
Core package for the RNA alignment generator.
"""

from .models import (
    RnaTriplet, DatasetMetadata, BulgeGraph, GraphNode,
    ModificationCounts, SampledModifications, ActionCounts, NodeType,
    ModificationType, classify_node,
    AlignmentLeaf, AlignmentResult, AlignmentMetadata,
)
from .rna_generator import RnaGenerator, BulgeGraphParser
from .modification_engine import ModificationEngine
from .bulge_graph_updater import BulgeGraphUpdater
from .alignment_generator import AlignmentDatasetGenerator

__all__ = [
    'RnaTriplet', 'DatasetMetadata', 'BulgeGraph', 'GraphNode',
    'ModificationCounts', 'SampledModifications', 'ActionCounts', 'NodeType',
    'ModificationType', 'classify_node', 'RnaGenerator', 
    'BulgeGraphParser', 'ModificationEngine', 'BulgeGraphUpdater',
    'AlignmentLeaf', 'AlignmentResult', 'AlignmentMetadata',
    'AlignmentDatasetGenerator'
]
