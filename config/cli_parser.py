"""
Command-line argument parser for the RNA triplet generator.
"""

import argparse
from enum import Enum
from typing import Optional


class SeqLenDistribution(Enum):
    """Distribution options for sequence length."""
    NORMAL = "norm"
    UNIFORM = "unif"


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="rna_generator",
        description="Generate RNA triplets (anchor/positive/negative) with modifications."
    )
    
    # Sequence Generation
    seq_group = parser.add_argument_group("Sequence Generation")
    seq_group.add_argument("--num_structures", type=int, default=100,
                          help="Number of structures (anchors) to generate")
    seq_group.add_argument("--seq_min_len", type=int, default=50,
                          help="Minimum sequence length")
    seq_group.add_argument("--seq_max_len", type=int, default=100,
                          help="Maximum sequence length")
    seq_group.add_argument("--seq_len_distribution", 
                          choices=[d.value for d in SeqLenDistribution],
                          default=SeqLenDistribution.UNIFORM.value,
                          help="Distribution of sequence lengths")
    seq_group.add_argument("--seq_len_mean", type=float, default=75.0,
                          help="Mean for normal distribution of sequence length")
    seq_group.add_argument("--seq_len_sd", type=float, default=10.0,
                          help="Standard deviation for normal distribution of sequence length")
    seq_group.add_argument("--neg_len_variation", type=int, default=0,
                          help="Maximum length variation for negative structures")
    
    # Stem Modifications
    stem_group = parser.add_argument_group("Stem Modifications")
    stem_group.add_argument("--n_stem_indels_min", type=int, default=0,
                           help="Minimum number of stem modification cycles per positive")
    stem_group.add_argument("--n_stem_indels_max", type=int, default=0,
                           help="Maximum number of stem modification cycles per positive")
    stem_group.add_argument("--n_stem_indels_mean", type=float, default=3.0,
                           help="Mean for truncated normal distribution of stem modifications")
    stem_group.add_argument("--n_stem_indels_sd", type=float, default=1.0,
                           help="Standard deviation for truncated normal distribution of stem modifications")
    stem_group.add_argument("--f_stem_indels_min", type=float, default=None,
                           help="Minimum fraction of stem positions to modify")
    stem_group.add_argument("--f_stem_indels_max", type=float, default=None,
                           help="Maximum fraction of stem positions to modify")
    stem_group.add_argument("--f_stem_indels_mean", type=float, default=None,
                           help="Mean for truncated normal distribution of stem modification fraction")
    stem_group.add_argument("--f_stem_indels_sd", type=float, default=None,
                           help="Standard deviation for truncated normal distribution of stem modification fraction")
    stem_group.add_argument("--stem_min_size", type=int, default=2,
                           help="Minimum stem size")
    stem_group.add_argument("--stem_max_size", type=int, default=10,
                           help="Maximum stem size")
    stem_group.add_argument("--same_stem_max_n_mod", type=int, default=1,
                           help="Maximum modifications per individual stem node")
    
    # Loop Modifications
    loop_group = parser.add_argument_group("Loop Modifications")
    
    # Hairpin loops
    loop_group.add_argument("--n_hloop_indels_min", type=int, default=0,
                           help="Minimum number of hairpin loop modification cycles per positive")
    loop_group.add_argument("--n_hloop_indels_max", type=int, default=0,
                           help="Maximum number of hairpin loop modification cycles per positive")
    loop_group.add_argument("--n_hloop_indels_mean", type=float, default=3.0,
                           help="Mean for truncated normal distribution of hairpin loop modifications")
    loop_group.add_argument("--n_hloop_indels_sd", type=float, default=1.0,
                           help="Standard deviation for truncated normal distribution of hairpin loop modifications")
    loop_group.add_argument("--f_hloop_indels_min", type=float, default=None,
                           help="Minimum fraction of hairpin loop positions to modify")
    loop_group.add_argument("--f_hloop_indels_max", type=float, default=None,
                           help="Maximum fraction of hairpin loop positions to modify")
    loop_group.add_argument("--f_hloop_indels_mean", type=float, default=None,
                           help="Mean for truncated normal distribution of hairpin loop modification fraction")
    loop_group.add_argument("--f_hloop_indels_sd", type=float, default=None,
                           help="Standard deviation for truncated normal distribution of hairpin loop modification fraction")
    loop_group.add_argument("--hloop_min_size", type=int, default=3,
                           help="Minimum hairpin loop size")
    loop_group.add_argument("--hloop_max_size", type=int, default=10,
                           help="Maximum hairpin loop size")
    loop_group.add_argument("--same_hloop_max_n_mod", type=int, default=1,
                           help="Maximum modifications per individual hairpin loop node")
    
    # Internal loops
    loop_group.add_argument("--n_iloop_indels_min", type=int, default=0,
                           help="Minimum number of internal loop modification cycles per positive")
    loop_group.add_argument("--n_iloop_indels_max", type=int, default=0,
                           help="Maximum number of internal loop modification cycles per positive")
    loop_group.add_argument("--n_iloop_indels_mean", type=float, default=3.0,
                           help="Mean for truncated normal distribution of internal loop modifications")
    loop_group.add_argument("--n_iloop_indels_sd", type=float, default=1.0,
                           help="Standard deviation for truncated normal distribution of internal loop modifications")
    loop_group.add_argument("--f_iloop_indels_min", type=float, default=None,
                           help="Minimum fraction of internal loop positions to modify")
    loop_group.add_argument("--f_iloop_indels_max", type=float, default=None,
                           help="Maximum fraction of internal loop positions to modify")
    loop_group.add_argument("--f_iloop_indels_mean", type=float, default=None,
                           help="Mean for truncated normal distribution of internal loop modification fraction")
    loop_group.add_argument("--f_iloop_indels_sd", type=float, default=None,
                           help="Standard deviation for truncated normal distribution of internal loop modification fraction")
    loop_group.add_argument("--iloop_min_size", type=int, default=2,
                           help="Minimum internal loop size")
    loop_group.add_argument("--iloop_max_size", type=int, default=10,
                           help="Maximum internal loop size")
    loop_group.add_argument("--same_iloop_max_n_mod", type=int, default=1,
                           help="Maximum modifications per individual internal loop node")
    
    # Bulge loops
    loop_group.add_argument("--n_bulge_indels_min", type=int, default=0,
                           help="Minimum number of bulge modification cycles per positive")
    loop_group.add_argument("--n_bulge_indels_max", type=int, default=0,
                           help="Maximum number of bulge modification cycles per positive")
    loop_group.add_argument("--n_bulge_indels_mean", type=float, default=3.0,
                           help="Mean for truncated normal distribution of bulge modifications")
    loop_group.add_argument("--n_bulge_indels_sd", type=float, default=1.0,
                           help="Standard deviation for truncated normal distribution of bulge modifications")
    loop_group.add_argument("--f_bulge_indels_min", type=float, default=None,
                           help="Minimum fraction of bulge positions to modify")
    loop_group.add_argument("--f_bulge_indels_max", type=float, default=None,
                           help="Maximum fraction of bulge positions to modify")
    loop_group.add_argument("--f_bulge_indels_mean", type=float, default=None,
                           help="Mean for truncated normal distribution of bulge modification fraction")
    loop_group.add_argument("--f_bulge_indels_sd", type=float, default=None,
                           help="Standard deviation for truncated normal distribution of bulge modification fraction")
    loop_group.add_argument("--bulge_min_size", type=int, default=1,
                           help="Minimum bulge loop size")
    loop_group.add_argument("--bulge_max_size", type=int, default=1,
                           help="Maximum bulge loop size")
    loop_group.add_argument("--same_bulge_max_n_mod", type=int, default=1,
                           help="Maximum modifications per individual bulge loop node")
    
    # Multi loops
    loop_group.add_argument("--n_mloop_indels_min", type=int, default=0,
                           help="Minimum number of multi loop modification cycles per positive")
    loop_group.add_argument("--n_mloop_indels_max", type=int, default=0,
                           help="Maximum number of multi loop modification cycles per positive")
    loop_group.add_argument("--n_mloop_indels_mean", type=float, default=3.0,
                           help="Mean for truncated normal distribution of multi loop modifications")
    loop_group.add_argument("--n_mloop_indels_sd", type=float, default=1.0,
                           help="Standard deviation for truncated normal distribution of multi loop modifications")
    loop_group.add_argument("--f_mloop_indels_min", type=float, default=None,
                           help="Minimum fraction of multi loop positions to modify")
    loop_group.add_argument("--f_mloop_indels_max", type=float, default=None,
                           help="Maximum fraction of multi loop positions to modify")
    loop_group.add_argument("--f_mloop_indels_mean", type=float, default=None,
                           help="Mean for truncated normal distribution of multi loop modification fraction")
    loop_group.add_argument("--f_mloop_indels_sd", type=float, default=None,
                           help="Standard deviation for truncated normal distribution of multi loop modification fraction")
    loop_group.add_argument("--mloop_min_size", type=int, default=2,
                           help="Minimum multi loop size")
    loop_group.add_argument("--mloop_max_size", type=int, default=15,
                           help="Maximum multi loop size")
    loop_group.add_argument("--same_mloop_max_n_mod", type=int, default=1,
                           help="Maximum modifications per individual multi loop node")
    
    # Appending Parameters
    append_group = parser.add_argument_group("Appending Parameters")
    append_group.add_argument("--appending_event_probability", type=float, default=0.3,
                             help="Probability that an appending event will occur for a triplet")
    append_group.add_argument("--both_sides_appending_probability", type=float, default=0.33,
                             help="Probability to append on both sides (otherwise left or right)")
    append_group.add_argument("--linker_min", type=int, default=2,
                             help="Minimum linker length (in bases) for appending event")
    append_group.add_argument("--linker_max", type=int, default=8,
                             help="Maximum linker length (in bases) for appending event")
    append_group.add_argument("--appending_size_factor", type=float, default=1.0,
                             help="Factor to multiply the anchor length for appended RNA length sampling")
    
    # Modification Normalization
    norm_group = parser.add_argument_group("Modification Normalization")
    norm_group.add_argument("--mod_normalization", action="store_true",
                           help="Enable normalization of modification counts based on anchor length")
    norm_group.add_argument("--normalization_len", type=int, default=100,
                           help="Normalization length to scale modifications")
    
    # Performance & Output
    perf_group = parser.add_argument_group("Performance & Output")
    perf_group.add_argument("--num_workers", type=int, default=4,
                           help="Number of parallel workers")
    perf_group.add_argument("--output_dir", type=str, default="output",
                           help="Directory to save output CSV, metadata, etc.")
    
    # Visualization
    viz_group = parser.add_argument_group("Visualization")
    viz_group.add_argument("--plot", action="store_true",
                          help="Generate structure plots")
    viz_group.add_argument("--num_plots", type=int, default=5,
                          help="Number of triplets to plot")
    
    # Dataset Splitting
    split_group = parser.add_argument_group("Dataset Splitting")
    split_group.add_argument("--split", action="store_true",
                            help="Enable splitting the dataset into train/val sets")
    split_group.add_argument("--train_fraction", type=float, default=0.8,
                            help="Fraction of data for training")
    split_group.add_argument("--val_fraction", type=float, default=0.2,
                            help="Fraction of data for validation")
    
    # Debug
    debug_group = parser.add_argument_group("Debug")
    debug_group.add_argument("--debug", action="store_true",
                            help="Enable debug logging")
    debug_group.add_argument("--timing_log", action="store_true",
                            help="Enable detailed timing logs")
    
    args = parser.parse_args()

    # Validate arguments
    if args.train_fraction + args.val_fraction > 1.0:
        parser.error("train_fraction + val_fraction cannot exceed 1.0")

    # Prevent mixing of fraction-based and count-based modification parameters
    def _check_mix(prefix: str, defaults: tuple):
        f_fields = [getattr(args, f"f_{prefix}_indels_min"),
                    getattr(args, f"f_{prefix}_indels_max"),
                    getattr(args, f"f_{prefix}_indels_mean"),
                    getattr(args, f"f_{prefix}_indels_sd")]
        n_fields = [getattr(args, f"n_{prefix}_indels_min"),
                    getattr(args, f"n_{prefix}_indels_max"),
                    getattr(args, f"n_{prefix}_indels_mean"),
                    getattr(args, f"n_{prefix}_indels_sd")]

        use_fraction = any(v is not None for v in f_fields)
        if use_fraction:
            # ensure all fraction parameters provided
            if not all(v is not None for v in f_fields):
                parser.error(f"All f_{prefix}_indels_* parameters must be provided when using fraction mode")
            # ensure no n parameters were altered from defaults
            if any(n != d for n, d in zip(n_fields, defaults)):
                parser.error(f"Cannot mix f_{prefix}_indels_* with n_{prefix}_indels_* parameters")

    _check_mix("stem", (0, 0, 3.0, 1.0))
    _check_mix("hloop", (0, 0, 3.0, 1.0))
    _check_mix("iloop", (0, 0, 3.0, 1.0))
    _check_mix("bulge", (0, 0, 3.0, 1.0))
    _check_mix("mloop", (0, 0, 3.0, 1.0))

    return args