#!/usr/bin/env python3
"""
Structure plotting utilities for RNA sequences and structures using rnartistcore.

This module provides functionality to visualize RNA secondary structures
using rnartistcore, similar to the functionality from the Rust version.
"""

import logging
import random
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional
import shutil

logger = logging.getLogger(__name__)


class RnartistCorePlotter:
    """Plots RNA secondary structures using rnartistcore."""
    
    def __init__(self):
        """Initialize the plotter and check for rnartistcore availability."""
        self.rnartistcore_available = self._check_rnartistcore()
        self.montage_available = self._check_montage()
        
    def _check_rnartistcore(self) -> bool:
        """Check if rnartistcore is available in the system."""
        try:
            result = subprocess.run(['rnartistcore', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info("rnartistcore found and available")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        logger.warning("rnartistcore not found. Structure plotting will be disabled.")
        return False
    
    def _check_montage(self) -> bool:
        """Check if ImageMagick montage is available for combining images."""
        try:
            result = subprocess.run(['montage', '-version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info("ImageMagick montage found and available")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        logger.warning("ImageMagick montage not found. Will create individual plots only.")
        return False
    
    def plot_triplet(self, triplet, output_path: Path, title: str = None) -> None:
        """
        Plot an RNA triplet (anchor, positive, negative) using rnartistcore.
        
        Args:
            triplet: RnaTriplet object containing sequences and structures
            output_path: Path to save the plot
            title: Optional title for the plot (not used in rnartistcore)
        """
        if not self.rnartistcore_available:
            logger.error("Cannot plot: rnartistcore not available")
            return
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Generate individual structure plots
            anchor_png = temp_path / f"triplet_{triplet.triplet_id}_anchor.png"
            positive_png = temp_path / f"triplet_{triplet.triplet_id}_positive.png"
            negative_png = temp_path / f"triplet_{triplet.triplet_id}_negative.png"
            
            # Create rnartistcore script for all three structures
            batch_script = self._create_triplet_batch_script(
                triplet, temp_path.resolve()
            )
            
            # Write batch script
            script_path = temp_path / "batch_plot.kts"
            with open(script_path, 'w') as f:
                f.write(batch_script)
            
            # Run rnartistcore
            try:
                result = subprocess.run(['rnartistcore', str(script_path)], 
                                      capture_output=True, text=True, timeout=60)
                if result.returncode != 0:
                    logger.error(f"rnartistcore failed: {result.stderr}")
                    return
            except subprocess.TimeoutExpired:
                logger.error("rnartistcore timed out")
                return
            
            # Check if all individual images were created
            if not (anchor_png.exists() and positive_png.exists() and negative_png.exists()):
                logger.error(f"One or more images for triplet {triplet.triplet_id} not found")
                return
            
            # Combine images using montage if available
            if self.montage_available:
                self._combine_images_with_montage_and_annotations(
                    [anchor_png, positive_png, negative_png], 
                    output_path, triplet
                )
            else:
                # If montage not available, just copy the anchor image as fallback
                shutil.copy2(anchor_png, output_path)
                logger.warning(f"Montage not available. Saved only anchor for triplet {triplet.triplet_id}")
        
        logger.debug(f"Saved triplet plot to {output_path}")
    
    def plot_structure(self, sequence: str, structure: str, output_path: Path, 
                      title: str = None, structure_name: str = "structure") -> None:
        """
        Plot a single RNA structure using rnartistcore.
        
        Args:
            sequence: RNA sequence string
            structure: Dot-bracket structure string
            output_path: Path to save the plot
            title: Optional title for the plot (not used in rnartistcore)
            structure_name: Name for the structure in rnartistcore
        """
        if not self.rnartistcore_available:
            logger.error("Cannot plot: rnartistcore not available")
            return
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create rnartistcore script
            script_content = self._create_single_structure_script(
                sequence, structure, temp_path.resolve(), structure_name
            )
            
            # Write script
            script_path = temp_path / "plot.kts"
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Run rnartistcore
            try:
                result = subprocess.run(['rnartistcore', str(script_path)], 
                                      capture_output=True, text=True, timeout=30)
                if result.returncode != 0:
                    logger.error(f"rnartistcore failed: {result.stderr}")
                    return
            except subprocess.TimeoutExpired:
                logger.error("rnartistcore timed out")
                return
            
            # Copy result to output path
            temp_png = temp_path / f"{structure_name}.png"
            if temp_png.exists():
                shutil.copy2(temp_png, output_path)
            else:
                logger.error(f"rnartistcore did not create expected output: {temp_png}")
        
        logger.debug(f"Saved structure plot to {output_path}")
    
    def plot_triplet_with_normalized_info(self, triplet, output_path: Path, title: str = None) -> None:
        """
        Plot an RNA triplet with normalized modification information displayed.
        
        Args:
            triplet: RnaTriplet object containing sequences and structures
            output_path: Path to save the plot
            title: Optional title for the plot (not used in rnartistcore)
        """
        if not self.rnartistcore_available:
            logger.error("Cannot plot: rnartistcore not available")
            return
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Generate individual structure plots
            anchor_png = temp_path / f"triplet_{triplet.triplet_id}_anchor.png"
            positive_png = temp_path / f"triplet_{triplet.triplet_id}_positive.png"
            negative_png = temp_path / f"triplet_{triplet.triplet_id}_negative.png"
            
            # Create rnartistcore script for all three structures
            batch_script = self._create_triplet_batch_script(
                triplet, temp_path.resolve()
            )
            
            # Write batch script
            script_path = temp_path / "batch_plot.kts"
            with open(script_path, 'w') as f:
                f.write(batch_script)
            
            # Run rnartistcore
            try:
                result = subprocess.run(['rnartistcore', str(script_path)], 
                                      capture_output=True, text=True, timeout=60)
                if result.returncode != 0:
                    logger.error(f"rnartistcore failed: {result.stderr}")
                    return
            except subprocess.TimeoutExpired:
                logger.error("rnartistcore timed out")
                return
            
            # Check if all individual images were created
            if not (anchor_png.exists() and positive_png.exists() and negative_png.exists()):
                logger.error(f"One or more images for triplet {triplet.triplet_id} not found")
                return
            
            # Combine images using montage if available
            if self.montage_available:
                self._combine_images_with_montage_and_normalized_annotations(
                    [anchor_png, positive_png, negative_png], 
                    output_path, triplet
                )
            else:
                # If montage not available, just copy the anchor image as fallback
                shutil.copy2(anchor_png, output_path)
                logger.warning(f"Montage not available. Saved only anchor for triplet {triplet.triplet_id}")
        
        logger.debug(f"Saved triplet plot with normalized info to {output_path}")
    
    def _create_triplet_batch_script(self, triplet, temp_dir: Path) -> str:
        """Create rnartistcore batch script for a triplet."""
        anchor_block = f'''rnartist {{
  png {{
    path = "{temp_dir}"
    width = 600.0
    height = 600.0
  }}
  ss {{
    bn {{
      value = "{triplet.anchor_structure}"
      seq = "{triplet.anchor_seq}"
      name = "triplet_{triplet.triplet_id}_anchor"
    }}
  }}
  theme {{
    details {{
      value = 5
    }}
    scheme {{
      value = "Pumpkin Vegas"
    }}
  }}
}}'''
        
        positive_block = f'''rnartist {{
  png {{
    path = "{temp_dir}"
    width = 600.0
    height = 600.0
  }}
  ss {{
    bn {{
      value = "{triplet.positive_structure}"
      seq = "{triplet.positive_seq}"
      name = "triplet_{triplet.triplet_id}_positive"
    }}
  }}
  theme {{
    details {{
      value = 5
    }}
    scheme {{
      value = "Pumpkin Vegas"
    }}
  }}
}}'''
        
        negative_block = f'''rnartist {{
  png {{
    path = "{temp_dir}"
    width = 600.0
    height = 600.0
  }}
  ss {{
    bn {{
      value = "{triplet.negative_structure}"
      seq = "{triplet.negative_seq}"
      name = "triplet_{triplet.triplet_id}_negative"
    }}
  }}
  theme {{
    details {{
      value = 5
    }}
    scheme {{
      value = "Pumpkin Vegas"
    }}
  }}
}}'''
        
        return f"{anchor_block}\n{positive_block}\n{negative_block}\n"
    
    def _create_single_structure_script(self, sequence: str, structure: str, 
                                      temp_dir: Path, name: str) -> str:
        """Create rnartistcore script for a single structure."""
        return f'''rnartist {{
  png {{
    path = "{temp_dir}"
    width = 600.0
    height = 600.0
  }}
  ss {{
    bn {{
      value = "{structure}"
      seq = "{sequence}"
      name = "{name}"
    }}
  }}
  theme {{
    details {{
      value = 5
    }}
    scheme {{
      value = "Pumpkin Vegas"
    }}
  }}
}}'''
    
    def _combine_images_with_montage(self, image_paths: List[Path], output_path: Path) -> None:
        """Combine multiple images into one using ImageMagick montage."""
        try:
            cmd = ['montage'] + [str(p) for p in image_paths] + [
                '-tile', '3x1',
                '-geometry', '+10+10',
                str(output_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                logger.error(f"montage failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            logger.error("montage timed out")
    
    def _combine_images_with_montage_and_annotations(self, image_paths: List[Path], 
                                                   output_path: Path, triplet) -> None:
        """Combine multiple images with titles and modification info using ImageMagick."""
        try:
            # Create annotated versions of each image with titles and modification info
            annotated_paths = []
            
            # Anchor image - no modifications
            anchor_annotated = image_paths[0].parent / f"annotated_{image_paths[0].name}"
            self._add_annotations_to_image(
                image_paths[0], anchor_annotated, 
                "Anchor", triplet, is_anchor=True
            )
            annotated_paths.append(anchor_annotated)
            
            # Positive image - with modifications
            positive_annotated = image_paths[1].parent / f"annotated_{image_paths[1].name}"
            self._add_annotations_to_image(
                image_paths[1], positive_annotated, 
                "Positive", triplet, is_anchor=False
            )
            annotated_paths.append(positive_annotated)
            
            # Negative image - no modifications
            negative_annotated = image_paths[2].parent / f"annotated_{image_paths[2].name}"
            self._add_annotations_to_image(
                image_paths[2], negative_annotated, 
                "Negative", triplet, is_anchor=True
            )
            annotated_paths.append(negative_annotated)
            
            # Combine annotated images
            cmd = ['montage'] + [str(p) for p in annotated_paths] + [
                '-tile', '3x1',
                '-geometry', '+20+20',
                '-background', 'white',
                str(output_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                logger.error(f"montage failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.error("montage timed out")
        except Exception as e:
            logger.error(f"Error in montage with annotations: {e}")
    
    def _add_annotations_to_image(self, input_path: Path, output_path: Path, 
                                structure_type: str, triplet, is_anchor: bool) -> None:
        """Add title and modification info to an image using ImageMagick convert."""
        try:
            # Create title text
            title = f"Triplet {triplet.triplet_id} - {structure_type}"
            
            # Create modification info text
            if is_anchor:
                mod_info = ""
            else:
                mod_lines = [
                    f"Total: {triplet.total_modifications}",
                    f"Stem: {triplet.stem_modifications}",
                    f"Hairpin: {triplet.hloop_modifications}",
                    f"Internal: {triplet.iloop_modifications}",
                    f"Bulge: {triplet.bulge_modifications}",
                    f"Multi: {triplet.mloop_modifications}"
                ]
                mod_info = " | ".join(mod_lines)
            
            # Use ImageMagick convert to add text annotations
            cmd = [
                'convert', str(input_path),
                '-background', 'white',
                '-fill', 'black',
                '-font', 'Arial',
                '-pointsize', '16',
                '-gravity', 'North',
                '-splice', '0x80',  # Add space at top for title
                '-annotate', '+0+10', title,
                '-pointsize', '12',
                '-gravity', 'South',
                '-splice', '0x60',  # Add space at bottom for mod info
                '-annotate', '+0+10', mod_info,
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                logger.error(f"convert failed: {result.stderr}")
                # Fallback: just copy the original
                shutil.copy2(input_path, output_path)
                
        except Exception as e:
            logger.error(f"Error adding annotations: {e}")
            # Fallback: just copy the original
            shutil.copy2(input_path, output_path)
    
    def _combine_images_with_montage_and_normalized_annotations(self, image_paths: List[Path], 
                                                             output_path: Path, triplet) -> None:
        """Combine multiple images with titles and normalized modification info using ImageMagick."""
        try:
            # Create annotated versions of each image with titles and normalized modification info
            annotated_paths = []
            
            # Anchor image - no modifications
            anchor_annotated = image_paths[0].parent / f"annotated_{image_paths[0].name}"
            self._add_normalized_annotations_to_image(
                image_paths[0], anchor_annotated, 
                "Anchor", triplet, is_anchor=True
            )
            annotated_paths.append(anchor_annotated)
            
            # Positive image - with modifications
            positive_annotated = image_paths[1].parent / f"annotated_{image_paths[1].name}"
            self._add_normalized_annotations_to_image(
                image_paths[1], positive_annotated, 
                "Positive", triplet, is_anchor=False
            )
            annotated_paths.append(positive_annotated)
            
            # Negative image - no modifications
            negative_annotated = image_paths[2].parent / f"annotated_{image_paths[2].name}"
            self._add_normalized_annotations_to_image(
                image_paths[2], negative_annotated, 
                "Negative", triplet, is_anchor=True
            )
            annotated_paths.append(negative_annotated)
            
            # Combine annotated images
            cmd = ['montage'] + [str(p) for p in annotated_paths] + [
                '-tile', '3x1',
                '-geometry', '+20+20',
                '-background', 'white',
                str(output_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                logger.error(f"montage failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.error("montage timed out")
        except Exception as e:
            logger.error(f"Error in montage with normalized annotations: {e}")

    def _add_normalized_annotations_to_image(self, input_path: Path, output_path: Path, 
                                           structure_type: str, triplet, is_anchor: bool) -> None:
        """Add title and normalized modification info to an image using ImageMagick convert."""
        try:
            # Create title text
            title = f"Triplet {triplet.triplet_id} - {structure_type}"
            
            # Create modification info text with normalized values
            if is_anchor:
                mod_info = "No modifications"
            else:
                # Format normalized modification with 2 decimal places as requested
                normalized_text = f"Normalized modification: {triplet.f_total_modifications:.2f}"
                mod_lines = [
                    normalized_text,
                    f"Total: {triplet.total_modifications}",
                    f"Stem: {triplet.stem_modifications}",
                    f"Hairpin: {triplet.hloop_modifications}",
                    f"Internal: {triplet.iloop_modifications}",
                    f"Bulge: {triplet.bulge_modifications}",
                    f"Multi: {triplet.mloop_modifications}"
                ]
                mod_info = " | ".join(mod_lines)
            
            # Use ImageMagick convert to add text annotations
            cmd = [
                'convert', str(input_path),
                '-background', 'white',
                '-fill', 'black',
                '-font', 'Arial',
                '-pointsize', '16',
                '-gravity', 'North',
                '-splice', '0x80',  # Add space at top for title
                '-annotate', '+0+10', title,
                '-pointsize', '12',
                '-gravity', 'South',
                '-splice', '0x80',  # Add more space at bottom for normalized mod info
                '-annotate', '+0+10', mod_info,
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                logger.error(f"convert failed: {result.stderr}")
                # Fallback: just copy the original
                shutil.copy2(input_path, output_path)
                
        except Exception as e:
            logger.error(f"Error adding normalized annotations: {e}")
            # Fallback: just copy the original
            shutil.copy2(input_path, output_path)


class StructurePlotManager:
    """Manages plotting of RNA structures for the dataset generator."""
    
    def __init__(self, args):
        """Initialize with command line arguments."""
        self.args = args
        self.plotter = RnartistCorePlotter()
        self.plot_dir = None
        
    def setup_plotting(self, output_dir: Path) -> None:
        """Setup the plotting directory."""
        if self.args.plot:
            self.plot_dir = output_dir / "plots"
            self.plot_dir.mkdir(exist_ok=True)
            logger.info(f"Plot directory created: {self.plot_dir}")
    
    def plot_triplets_stratified(self, triplets: List, num_plots: Optional[int] = None) -> None:
        """
        Plot triplets using stratified sampling based on f_total_modifications distribution.
        
        Args:
            triplets: List of RnaTriplet objects
            num_plots: Number of triplets to plot (defaults to args.num_plots)
        """
        if not self.args.plot or not self.plot_dir:
            return
        
        if not self.plotter.rnartistcore_available:
            logger.error("Cannot plot: rnartistcore not available")
            return
        
        if not triplets:
            logger.warning("No triplets to plot")
            return
        
        num_plots = num_plots or self.args.num_plots
        num_plots = min(num_plots, len(triplets))
        
        # Calculate percentile size based on number of plots
        percentile_size = 100.0 / num_plots  # e.g., 50 plots = 2%, 100 plots = 1%, 200 plots = 0.5%
        
        logger.info(f"Plotting {num_plots} triplets using stratified sampling based on f_total_modifications...")
        logger.info(f"Using {percentile_size:.1f}% percentiles for sampling")
        
        # Sort triplets by f_total_modifications (highest to lowest)
        sorted_triplets = sorted(triplets, key=lambda t: t.f_total_modifications, reverse=True)
        
        selected_triplets = []
        
        for i in range(num_plots):
            # Calculate which percentile this sample should come from
            # Sample i comes from the i-th percentile
            start_percentile = i * percentile_size / 100.0  # Convert to fraction
            end_percentile = (i + 1) * percentile_size / 100.0
            
            # Convert percentiles to indices
            start_idx = int(start_percentile * len(sorted_triplets))
            end_idx = int(end_percentile * len(sorted_triplets))
            
            # Ensure we don't go out of bounds
            start_idx = min(start_idx, len(sorted_triplets) - 1)
            end_idx = min(end_idx, len(sorted_triplets))
            
            # If start_idx == end_idx, we're at the end - just take the last available triplet
            if start_idx >= end_idx:
                if start_idx < len(sorted_triplets):
                    selected_triplet = sorted_triplets[start_idx]
                else:
                    selected_triplet = sorted_triplets[-1]
            else:
                # Randomly select one triplet from this percentile range
                selected_triplet = random.choice(sorted_triplets[start_idx:end_idx])
            
            selected_triplets.append(selected_triplet)
        
        logger.info(f"Selected triplets from distribution: f_total_modifications range from "
                   f"{selected_triplets[-1].f_total_modifications:.4f} to {selected_triplets[0].f_total_modifications:.4f}")
        
        # Plot each selected triplet
        for i, triplet in enumerate(selected_triplets):
            # Format f_total_modifications for filename (4 decimal places, remove leading zero)
            f_mod_str = f"{triplet.f_total_modifications:.4f}"[1:]  # Remove "0." -> ".1234"
            f_mod_str = f_mod_str.replace(".", "")  # ".1234" -> "1234"
            
            output_path = self.plot_dir / f"f_{f_mod_str}_triplet_{triplet.triplet_id}.png"
            title = f"RNA Triplet {triplet.triplet_id} - f_total_modifications: {triplet.f_total_modifications:.2f}"
            
            try:
                self.plotter.plot_triplet_with_normalized_info(triplet, output_path, title)
                
                if (i + 1) % 5 == 0 or i == len(selected_triplets) - 1:
                    logger.info(f"Plotted {i + 1}/{len(selected_triplets)} stratified triplets")
                    
            except Exception as e:
                logger.error(f"Failed to plot triplet {triplet.triplet_id}: {e}")
                continue
        
        logger.info(f"Stratified plotting completed. Plots saved to: {self.plot_dir}")

    def plot_triplets(self, triplets: List, num_plots: Optional[int] = None) -> None:
        """
        Plot a selection of triplets from the dataset.
        
        Args:
            triplets: List of RnaTriplet objects
            num_plots: Number of triplets to plot (defaults to args.num_plots)
        """
        if not self.args.plot or not self.plot_dir:
            return
        
        if not self.plotter.rnartistcore_available:
            logger.error("Cannot plot: rnartistcore not available")
            return
        
        if not triplets:
            logger.warning("No triplets to plot")
            return
        
        num_plots = num_plots or self.args.num_plots
        num_plots = min(num_plots, len(triplets))
        
        logger.info(f"Plotting {num_plots} triplets using rnartistcore...")
        
        # Select triplets to plot (random sampling if more than requested)
        if len(triplets) <= num_plots:
            selected_triplets = triplets
        else:
            selected_triplets = random.sample(triplets, num_plots)
        
        for i, triplet in enumerate(selected_triplets):
            output_path = self.plot_dir / f"triplet_{triplet.triplet_id:04d}.png"
            title = f"RNA Triplet {triplet.triplet_id}"
            
            try:
                self.plotter.plot_triplet(triplet, output_path, title)
                
                if (i + 1) % 5 == 0 or i == len(selected_triplets) - 1:
                    logger.info(f"Plotted {i + 1}/{len(selected_triplets)} triplets")
                    
            except Exception as e:
                logger.error(f"Failed to plot triplet {triplet.triplet_id}: {e}")
                continue
        
        logger.info(f"Plotting completed. Plots saved to: {self.plot_dir}")

    def plot_sample_structures(self, triplets: List, sample_size: int = 5) -> None:
        """
        Plot individual structures from a sample of triplets using rnartistcore.
        
        Args:
            triplets: List of RnaTriplet objects
            sample_size: Number of sample structures to plot
        """
        if not self.args.plot or not self.plot_dir:
            return
        
        if not self.plotter.rnartistcore_available:
            logger.error("Cannot plot individual structures: rnartistcore not available")
            return
        
        if not triplets:
            return
        
        sample_size = min(sample_size, len(triplets))
        sampled_triplets = random.sample(triplets, sample_size)
        
        structures_dir = self.plot_dir / "individual_structures"
        structures_dir.mkdir(exist_ok=True)
        
        for triplet in sampled_triplets:
            base_name = f"triplet_{triplet.triplet_id:04d}"
            
            # Plot anchor
            self.plotter.plot_structure(
                triplet.anchor_seq, triplet.anchor_structure,
                structures_dir / f"{base_name}_anchor.png",
                f"Anchor - Triplet {triplet.triplet_id}",
                f"{base_name}_anchor"
            )
            
            # Plot positive
            self.plotter.plot_structure(
                triplet.positive_seq, triplet.positive_structure,
                structures_dir / f"{base_name}_positive.png",
                f"Positive - Triplet {triplet.triplet_id} ({triplet.total_modifications} mods)",
                f"{base_name}_positive"
            )
            
            # Plot negative
            self.plotter.plot_structure(
                triplet.negative_seq, triplet.negative_structure,
                structures_dir / f"{base_name}_negative.png",
                f"Negative - Triplet {triplet.triplet_id}",
                f"{base_name}_negative"
            )
        
        logger.info(f"Individual structure plots saved to: {structures_dir}")