"""
Logging utilities for the RNA triplet generator.
"""

import logging
import sys
from pathlib import Path


def setup_logging(debug: bool = False, log_file: str = None) -> None:
    """
    Setup logging configuration for the application.
    
    Args:
        debug: Enable debug level logging
        log_file: Optional log file path
    """
    # Configure logging level
    level = logging.DEBUG if debug else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels for external libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    # Suppress forgi logging messages
    logging.getLogger('forgi').setLevel(logging.WARNING)
    logging.getLogger('forgi.graph').setLevel(logging.WARNING)
    logging.getLogger('forgi.graph.bulge_graph').setLevel(logging.WARNING)
    logging.getLogger('forgi.graph._cofold').setLevel(logging.WARNING)