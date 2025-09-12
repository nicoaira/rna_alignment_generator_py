#!/usr/bin/env python3
"""
RNA Secondary Structure Alignment Generator - Python Version

Simulates evolution from ancestral sequences across a binary tree with
structure-aware mutations and outputs alignments of the leaf sequences.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from config.cli_parser import parse_arguments
from core.alignment_generator import AlignmentDatasetGenerator
from utils.logger import setup_logging
from utils.output_handler import OutputHandler
# Plotting utilities not used in alignment mode


def main() -> int:
    """Main entry point for the RNA alignment generator."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Setup logging
        setup_logging(debug=args.debug)
        logger = logging.getLogger(__name__)
        
        logger.info("Starting RNA Alignment Generator (Python version)")
        logger.info(f"Generating {args.num_alignments} alignments with {args.num_cycles} cycles")
        logger.info(f"Output directory: {args.output_dir}")
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Plotting disabled in alignment mode
        
        # Initialize alignment generator
        generator = AlignmentDatasetGenerator(args)

        # Generate alignments
        logger.info("Generating alignments...")
        alignments = generator.generate_alignments()

        # Handle output
        output_handler = OutputHandler(args)
        output_handler.save_alignments(alignments, output_dir)
        
        logger.info("Alignment generation completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
