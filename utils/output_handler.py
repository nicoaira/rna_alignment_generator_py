"""
Output handling utilities for saving alignments and metadata.
"""

import csv
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class OutputHandler:
    """Handles saving datasets and metadata to various formats."""
    
    def __init__(self, args):
        """Initialize the output handler with configuration."""
        self.args = args
    
    def save_alignments(self, alignments: List, output_dir: Path) -> None:
        """
        Save alignments (multi-FASTA) and metadata.
        
        Args:
            alignments: List of AlignmentResult
            output_dir: Directory to save files
        """
        n = len(alignments)
        logger.info(f"Saving {n} alignments to {output_dir}")

        # Create metadata
        from core.models import AlignmentMetadata
        metadata = AlignmentMetadata.create(self.args, n)

        # Ensure subdir exists
        align_dir = output_dir / "alignments"
        align_dir.mkdir(exist_ok=True)

        export_dataset = getattr(self.args, "prepare_dataset", False)
        dataset_rows: List[Dict[str, object]] = []
        conserved_mapping: Dict[str, Dict[str, Dict[int, int]]] = {}
        sequence_counter = 0

        # Save each alignment as FASTA with gaps and Stockholm with sequence+structure
        for res in alignments:
            path = align_dir / f"alignment_{res.alignment_id:05d}.fasta"
            with open(path, 'w') as f:
                for leaf in res.leaves:
                    f.write(f">{leaf.leaf_id}\n")
                    f.write(leaf.aligned_sequence + "\n")

            # Stockholm format file
            sto_path = align_dir / f"alignment_{res.alignment_id:05d}.sto"
            with open(sto_path, 'w') as sf:
                sf.write("# STOCKHOLM 1.0\n")
                # Write aligned sequences
                for leaf in res.leaves:
                    sf.write(f"{leaf.leaf_id}\t{leaf.aligned_sequence}\n")
                # Write per-sequence structure annotations
                for leaf in res.leaves:
                    sf.write(f"#=GR {leaf.leaf_id} SS {leaf.aligned_structure}\n")
                # Write per-column conservation mask from root commitment
                if hasattr(res, 'gc_conservation') and res.gc_conservation:
                    sf.write(f"#=GC CONS {res.gc_conservation}\n")
                sf.write("//\n")

            if export_dataset:
                alignment_label = self._format_alignment_label(res.alignment_id)
                alignment_mapping: Dict[str, Dict[int, int]] = {}

                for leaf in res.leaves:
                    sequence_counter += 1
                    binary_code = leaf.path if leaf.path else "root"
                    dataset_rows.append({
                        'alignment_id': alignment_label,
                        'sequence_id': sequence_counter,
                        'binary_code': binary_code,
                        'sequence': leaf.sequence,
                        'structure': leaf.structure,
                    })

                    if res.gc_conservation:
                        conserved_positions = self._map_conserved_positions(leaf, res.gc_conservation)
                        if conserved_positions:
                            alignment_mapping[f"rna_{sequence_counter}"] = conserved_positions

                if alignment_mapping:
                    conserved_mapping[alignment_label] = alignment_mapping

        # Save metadata
        self._save_metadata(metadata, output_dir / "metadata.json")

        if export_dataset:
            dataset_path = output_dir / "alignment_dataset.tsv"
            mapping_path = output_dir / "alignment_conserved_mapping.json"
            self._write_alignment_dataset(dataset_rows, dataset_path)
            self._write_alignment_mapping(conserved_mapping, mapping_path)
        logger.info("Alignments saved successfully!")
    
    # Triplet-specific methods retained for backwards compatibility but unused now
    def _save_single_dataset(self, triplets: List, 
                           metadata, output_dir: Path) -> None:
        """Save dataset as a single file."""
        csv_path = output_dir / "rna_triplets.csv"
        self._save_triplets_csv(triplets, csv_path, metadata)
    
    def _save_split_dataset(self, triplets: List, 
                          metadata, output_dir: Path) -> None:
        """Save dataset split into train and validation sets."""
        # Shuffle triplets
        shuffled_triplets = triplets.copy()
        random.shuffle(shuffled_triplets)
        
        # Calculate split sizes
        total_size = len(shuffled_triplets)
        train_size = int(total_size * self.args.train_fraction)
        
        # Split the data
        train_triplets = shuffled_triplets[:train_size]
        val_triplets = shuffled_triplets[train_size:]
        
        logger.info(f"Split dataset: {len(train_triplets)} train, {len(val_triplets)} validation")
        
        # Save splits
        train_path = output_dir / "rna_triplets_train.csv"
        val_path = output_dir / "rna_triplets_val.csv"
        full_path = output_dir / "rna_triplets.csv"
        
        self._save_triplets_csv(train_triplets, train_path, metadata)
        self._save_triplets_csv(val_triplets, val_path, metadata)
        self._save_triplets_csv(triplets, full_path, metadata)  # Also save full dataset
    
    def _save_triplets_csv(self, triplets: List, 
                          file_path: Path, metadata) -> None:
        """
        Save triplets to CSV file with metadata header.
        
        Args:
            triplets: List of triplets to save
            file_path: Output file path
            metadata: Dataset metadata
        """
        logger.info(f"Saving {len(triplets)} triplets to {file_path}")
        
        with open(file_path, 'w', newline='') as csvfile:
            # Write metadata as comment
            metadata_json = json.dumps(metadata.__dict__, indent=None)
            csvfile.write(f"# Metadata: {metadata_json}\n")
            
            # Write CSV data
            fieldnames = [
                'triplet_id', 'anchor_seq', 'anchor_structure',
                'positive_seq', 'positive_structure', 'negative_seq', 'negative_structure',
                'total_modifications', 'stem_modifications', 'hloop_modifications',
                'iloop_modifications', 'bulge_modifications', 'mloop_modifications',
                'total_len_stem', 'total_len_hloop', 'total_len_iloop',
                'total_len_bulge', 'total_len_mloop',
                'anchor_seq_len', 'positive_seq_len', 'negative_seq_len',
                'stem_insertions', 'stem_deletions',
                'hloop_insertions', 'hloop_deletions',
                'iloop_insertions', 'iloop_deletions',
                'bulge_insertions', 'bulge_deletions',
                'mloop_insertions', 'mloop_deletions',
                'total_insertions', 'total_deletions',
                'f_stem_modifications', 'f_stem_insertions', 'f_stem_deletions',
                'f_hloop_modifications', 'f_hloop_insertions', 'f_hloop_deletions',
                'f_iloop_modifications', 'f_iloop_insertions', 'f_iloop_deletions',
                'f_bulge_modifications', 'f_bulge_insertions', 'f_bulge_deletions',
                'f_mloop_modifications', 'f_mloop_insertions', 'f_mloop_deletions',
                'f_total_modifications', 'f_total_insertions', 'f_total_deletions'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for triplet in triplets:
                writer.writerow({
                    'triplet_id': triplet.triplet_id,
                    'anchor_seq': triplet.anchor_seq,
                    'anchor_structure': triplet.anchor_structure,
                    'positive_seq': triplet.positive_seq,
                    'positive_structure': triplet.positive_structure,
                    'negative_seq': triplet.negative_seq,
                    'negative_structure': triplet.negative_structure,
                    'total_modifications': triplet.total_modifications,
                    'stem_modifications': triplet.stem_modifications,
                    'hloop_modifications': triplet.hloop_modifications,
                    'iloop_modifications': triplet.iloop_modifications,
                    'bulge_modifications': triplet.bulge_modifications,
                    'mloop_modifications': triplet.mloop_modifications,
                    'total_len_stem': triplet.total_len_stem,
                    'total_len_hloop': triplet.total_len_hloop,
                    'total_len_iloop': triplet.total_len_iloop,
                    'total_len_bulge': triplet.total_len_bulge,
                    'total_len_mloop': triplet.total_len_mloop,
                    'anchor_seq_len': triplet.anchor_seq_len,
                    'positive_seq_len': triplet.positive_seq_len,
                    'negative_seq_len': triplet.negative_seq_len,
                    'stem_insertions': triplet.stem_insertions,
                    'stem_deletions': triplet.stem_deletions,
                    'hloop_insertions': triplet.hloop_insertions,
                    'hloop_deletions': triplet.hloop_deletions,
                    'iloop_insertions': triplet.iloop_insertions,
                    'iloop_deletions': triplet.iloop_deletions,
                    'bulge_insertions': triplet.bulge_insertions,
                    'bulge_deletions': triplet.bulge_deletions,
                    'mloop_insertions': triplet.mloop_insertions,
                    'mloop_deletions': triplet.mloop_deletions,
                    'total_insertions': triplet.total_insertions,
                    'total_deletions': triplet.total_deletions,
                    'f_stem_modifications': triplet.f_stem_modifications,
                    'f_stem_insertions': triplet.f_stem_insertions,
                    'f_stem_deletions': triplet.f_stem_deletions,
                    'f_hloop_modifications': triplet.f_hloop_modifications,
                    'f_hloop_insertions': triplet.f_hloop_insertions,
                    'f_hloop_deletions': triplet.f_hloop_deletions,
                    'f_iloop_modifications': triplet.f_iloop_modifications,
                    'f_iloop_insertions': triplet.f_iloop_insertions,
                    'f_iloop_deletions': triplet.f_iloop_deletions,
                    'f_bulge_modifications': triplet.f_bulge_modifications,
                    'f_bulge_insertions': triplet.f_bulge_insertions,
                    'f_bulge_deletions': triplet.f_bulge_deletions,
                    'f_mloop_modifications': triplet.f_mloop_modifications,
                    'f_mloop_insertions': triplet.f_mloop_insertions,
                    'f_mloop_deletions': triplet.f_mloop_deletions,
                    'f_total_modifications': triplet.f_total_modifications,
                    'f_total_insertions': triplet.f_total_insertions,
                    'f_total_deletions': triplet.f_total_deletions
                })
    
    def _save_metadata(self, metadata, file_path: Path) -> None:
        """Save metadata to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(metadata.__dict__, f, indent=2)
        
        logger.info(f"Metadata saved to {file_path}")

    def _format_alignment_label(self, alignment_id: int) -> str:
        """Return a human-friendly alignment label starting at 1."""
        return f"alignment_{alignment_id + 1}"

    def _map_conserved_positions(self, leaf, conservation: str) -> Dict[int, int]:
        """Map conserved alignment columns to sequence positions for a leaf."""
        mapping: Dict[int, int] = {}
        seq_pos = 0
        limit = min(len(leaf.aligned_sequence), len(conservation))
        for idx in range(limit):
            base = leaf.aligned_sequence[idx]
            if base != '-':
                seq_pos += 1
                if conservation[idx] == '1':
                    mapping[idx + 1] = seq_pos
            else:
                # skip gaps even if column marked conserved
                continue
        return mapping

    def _write_alignment_dataset(self, rows: List[Dict[str, object]], file_path: Path) -> None:
        """Write the alignment dataset rows to a TSV file."""
        logger.info(f"Saving alignment dataset TSV to {file_path}")
        fieldnames = ['alignment_id', 'sequence_id', 'binary_code', 'sequence', 'structure']
        with open(file_path, 'w', newline='') as tsv_file:
            writer = csv.DictWriter(tsv_file, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _write_alignment_mapping(self, mapping: Dict[str, Dict[str, Dict[int, int]]], file_path: Path) -> None:
        """Write conserved alignment position mappings to JSON."""
        logger.info(f"Saving conserved position mapping JSON to {file_path}")
        with open(file_path, 'w') as json_file:
            json.dump(mapping, json_file, indent=2)


class DatasetAnalyzer:
    """Analyze and generate statistics for RNA triplet datasets."""
    
    @staticmethod
    def analyze_dataset(triplets: List) -> dict:
        """
        Analyze a dataset and return statistics.
        
        Args:
            triplets: List of RNA triplets
            
        Returns:
            Dictionary containing dataset statistics
        """
        if not triplets:
            return {}
        
        # Basic statistics
        num_triplets = len(triplets)
        
        # Sequence length statistics
        anchor_lengths = [len(t.anchor_seq) for t in triplets]
        positive_lengths = [len(t.positive_seq) for t in triplets]
        negative_lengths = [len(t.negative_seq) for t in triplets]
        
        # Modification statistics
        total_mods = [t.total_modifications for t in triplets]
        stem_mods = [t.stem_modifications for t in triplets]
        hloop_mods = [t.hloop_modifications for t in triplets]
        iloop_mods = [t.iloop_modifications for t in triplets]
        bulge_mods = [t.bulge_modifications for t in triplets]
        mloop_mods = [t.mloop_modifications for t in triplets]
        
        stats = {
            'num_triplets': num_triplets,
            'sequence_lengths': {
                'anchor': {
                    'mean': sum(anchor_lengths) / len(anchor_lengths),
                    'min': min(anchor_lengths),
                    'max': max(anchor_lengths)
                },
                'positive': {
                    'mean': sum(positive_lengths) / len(positive_lengths),
                    'min': min(positive_lengths),
                    'max': max(positive_lengths)
                },
                'negative': {
                    'mean': sum(negative_lengths) / len(negative_lengths),
                    'min': min(negative_lengths),
                    'max': max(negative_lengths)
                }
            },
            'modifications': {
                'total': {
                    'mean': sum(total_mods) / len(total_mods),
                    'min': min(total_mods),
                    'max': max(total_mods)
                },
                'stem': {
                    'mean': sum(stem_mods) / len(stem_mods),
                    'min': min(stem_mods),
                    'max': max(stem_mods)
                },
                'hairpin': {
                    'mean': sum(hloop_mods) / len(hloop_mods),
                    'min': min(hloop_mods),
                    'max': max(hloop_mods)
                },
                'internal': {
                    'mean': sum(iloop_mods) / len(iloop_mods),
                    'min': min(iloop_mods),
                    'max': max(iloop_mods)
                },
                'bulge': {
                    'mean': sum(bulge_mods) / len(bulge_mods),
                    'min': min(bulge_mods),
                    'max': max(bulge_mods)
                },
                'multi': {
                    'mean': sum(mloop_mods) / len(mloop_mods),
                    'min': min(mloop_mods),
                    'max': max(mloop_mods)
                }
            }
        }
        
        return stats
    
    @staticmethod
    def print_statistics(stats: dict) -> None:
        """Print dataset statistics in a readable format."""
        print("\n=== Dataset Statistics ===")
        print(f"Number of triplets: {stats['num_triplets']}")
        
        print("\nSequence Lengths:")
        for seq_type, length_stats in stats['sequence_lengths'].items():
            print(f"  {seq_type.title()}: mean={length_stats['mean']:.1f}, "
                  f"min={length_stats['min']}, max={length_stats['max']}")
        
        print("\nModifications:")
        for mod_type, mod_stats in stats['modifications'].items():
            print(f"  {mod_type.title()}: mean={mod_stats['mean']:.2f}, "
                  f"min={mod_stats['min']}, max={mod_stats['max']}")
        print("=" * 30)
