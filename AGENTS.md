# RNA Triplet Generator - Agent Documentation

## Package Overview

The RNA Triplet Generator is a modular Python package for creating machine learning datasets consisting of RNA sequence triplets (anchor/positive/negative) with controlled structural modifications. This package is designed for training and evaluating RNA structure prediction models.

### Core Functionality
- **RNA Sequence Generation**: Creates random RNA sequences with configurable length distributions
- **Structure Folding**: Integrates with ViennaRNA for secondary structure prediction using dot-bracket notation
- **Structural Modifications**: Applies controlled insertions/deletions to specific RNA structural elements
- **Negative Sample Generation**: Creates negative samples through dinucleotide shuffling while preserving composition
- **Dataset Generation**: Orchestrates the complete triplet generation pipeline with parallel processing support

## Package Structure

```
rna_triplet_generator_py/
├── main.py                     # Entry point with CLI interface
├── requirements.txt            # Python dependencies
├── setup.py                   # Package installation script
├── config/                    # Configuration and argument parsing
│   ├── __init__.py
│   └── cli_parser.py          # Command-line argument definitions
├── core/                      # Core functionality modules
│   ├── __init__.py
│   ├── models.py              # Data structures and enums
│   ├── rna_generator.py       # RNA sequence generation and folding
│   ├── modification_engine.py # Structural modification algorithms
│   ├── negative_generator.py  # Negative sample creation
│   └── dataset_generator.py   # Main orchestration logic
└── utils/                     # Utility modules
    ├── __init__.py
    ├── logger.py              # Logging configuration
    └── output_handler.py      # Data export and analysis
```

## Core Data Models

### NodeType Enum
Classifies RNA structural elements:
- `STEM`: Base-paired regions forming stems
- `HAIRPIN`: Terminal loops at stem ends
- `INTERNAL`: Interior loops with unpaired bases on both strands
- `BULGE`: Interior loops with unpaired bases on one strand
- `MULTI`: Multi-branched loops connecting multiple stems
- `FIVE_PRIME`/`THREE_PRIME`: Terminal regions
- `UNKNOWN`: Unclassified elements

### ModificationType Enum
Types of structural modifications:
- `INSERT`: Single base insertion
- `DELETE`: Single base deletion
- `INSERT_PAIR`: Complementary base pair insertion (stems)
- `DELETE_PAIR`: Complementary base pair deletion (stems)

### RnaTriplet Data Structure
```python
@dataclass
class RnaTriplet:
    triplet_id: int
    anchor_seq: str                # Original RNA sequence
    anchor_structure: str          # Original dot-bracket structure
    positive_seq: str              # Modified sequence
    positive_structure: str        # Modified structure
    negative_seq: str              # Shuffled sequence
    negative_structure: str        # Shuffled structure
    total_modifications: int       # Total modification count
    stem_modifications: int        # Stem-specific modifications
    hloop_modifications: int       # Hairpin loop modifications
    iloop_modifications: int       # Internal loop modifications
    bulge_modifications: int       # Bulge modifications
    mloop_modifications: int       # Multi-loop modifications
```

### BulgeGraph Structure
Represents RNA secondary structure as a graph of structural elements:
```python
@dataclass
class BulgeGraph:
    elements: Dict[str, GraphNode]  # Node name -> GraphNode mapping
    
    def node_mapping(self) -> Dict[str, List[int]]
    def get_nodes_by_type(self, node_type: NodeType) -> List[Tuple[str, List[int]]]
```

## Core Modules

### 1. models.py
**Purpose**: Defines data structures and enums for the entire package.

**Key Classes**:
- `RnaTriplet`: Complete triplet data structure
- `BulgeGraph`: RNA secondary structure representation
- `ModificationCounts`: Tracking applied modifications
- `SampledModifications`: Modification count sampling
- `DatasetMetadata`: Dataset generation metadata

**Key Functions**:
- `classify_node(node_name, coords)`: Classifies structural elements by name and coordinate count

### 2. rna_generator.py
**Purpose**: RNA sequence generation and structure folding.

**Key Classes**:
- `RnaGenerator`: Random RNA sequence generation with configurable distributions
- `BulgeGraphParser`: Integration with forgi library for structure parsing

**Key Methods**:
- `generate_sequence(length)`: Creates random RNA sequences
- `fold_sequence(sequence)`: Predicts secondary structure using ViennaRNA
- `parse_structure(structure)`: Converts dot-bracket to bulge graph representation

### 3. modification_engine.py
**Purpose**: Applies controlled structural modifications to RNA sequences.

**Key Class**: `ModificationEngine`

**Key Methods**:
- `apply_modifications()`: Main modification pipeline
- `_modify_stems()`: Applies insertions/deletions to stem regions
- `_modify_loops()`: Applies modifications to loop regions
- `_insert_stem_pair()`: Inserts complementary base pairs at random positions
- `_delete_stem_pair()`: Deletes complementary base pairs at random positions
- `_insert_loop_base()`: Inserts bases in loop regions
- `_delete_loop_base()`: Deletes bases from loop regions

**Modification Strategy**:
- **Stems**: Random position selection with proper base pairing maintenance
- **Loops**: Random position selection within loop boundaries
- **Size Constraints**: Respects minimum/maximum size limits for each element type
- **Modification Limits**: Limits number of modifications per structural element

### 4. negative_generator.py
**Purpose**: Creates negative samples through sequence shuffling.

**Key Classes**:
- `NegativeSampleGenerator`: Dinucleotide shuffling preserving composition
- `AppendingEngine`: Optional sequence extension with linkers

**Key Methods**:
- `generate_negative()`: Creates shuffled negative sample
- `shuffle_dinucleotides()`: Preserves dinucleotide composition while randomizing sequence
- `apply_appending()`: Optionally extends sequences with random linkers

### 5. dataset_generator.py
**Purpose**: Main orchestration of the triplet generation pipeline.

**Key Class**: `DatasetGenerator`

**Key Methods**:
- `generate_dataset()`: Main generation pipeline
- `generate_triplet()`: Creates single triplet
- `_generate_sequential()`: Sequential generation mode
- `_generate_parallel()`: Parallel generation using multiprocessing

## Configuration Parameters

### Sequence Generation
- `seq_min_len`/`seq_max_len`: Sequence length range (default: 20-100)
- `seq_len_distribution`: Length distribution type (`unif`, `norm`)
- `seq_len_mean`/`seq_len_sd`: Normal distribution parameters

### Modification Parameters (per structural element)
- `n_*_indels_min`/`max`: Modification count ranges
- `n_*_indels_mean`/`sd`: Normal distribution parameters for modification sampling
- `*_min_size`/`*_max_size`: Size constraints for structural elements
- `same_*_max_n_mod`: Maximum modifications per individual element

### Processing Options
- `num_workers`: Parallel processing workers (default: 1)
- `batch_size`: Batch size for parallel processing (default: 100)
- `mod_normalization`: Scale modifications by sequence length
- `normalization_len`: Reference length for scaling (default: 100)

### Output Options
- `output_dir`: Output directory path
- `split`: Enable train/validation splitting
- `train_fraction`/`val_fraction`: Dataset split ratios

## Usage Examples

### Basic Usage
```bash
python main.py --num_structures 1000
```

### Advanced Configuration
```bash
python main.py \
    --num_structures 10000 \
    --seq_min_len 50 \
    --seq_max_len 200 \
    --n_stem_indels_min 1 \
    --n_stem_indels_max 3 \
    --mod_normalization \
    --num_workers 8 \
    --split \
    --debug
```

### Programmatic Usage
```python
from core import DatasetGenerator
from config.cli_parser import parse_arguments

# Parse command line arguments or create custom config
args = parse_arguments()

# Create and run generator
generator = DatasetGenerator(args)
triplets, metadata = generator.generate_dataset()
```

## Output Format

### CSV Structure
The generated dataset is saved as CSV with columns:
- `triplet_id`: Unique identifier
- `anchor_seq`, `anchor_structure`: Original sequence and structure
- `positive_seq`, `positive_structure`: Modified sequence and structure  
- `negative_seq`, `negative_structure`: Shuffled sequence and structure
- `total_modifications`: Total modification count
- `*_modifications`: Per-element modification counts

### Metadata
JSON metadata includes:
- Generation parameters
- Dataset statistics
- Timestamp and run ID
- Element-wise modification distributions

## Dependencies

### Required
- **Python 3.8+**: Core runtime
- **ViennaRNA**: RNA folding (system dependency)
- **forgi>=2.0.0**: Bulge graph parsing
- **numpy>=1.21.0**: Numerical computations
- **scipy>=1.7.0**: Statistical distributions
- **tqdm>=4.62.0**: Progress bars

### Optional
- **pandas>=1.3.0**: Data analysis
- **matplotlib>=3.5.0**: Plotting
- **pytest>=6.0.0**: Testing

## Installation

1. **Install ViennaRNA**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install vienna-rna
   
   # macOS
   brew install viennarna
   
   # Conda
   conda install -c bioconda viennarna
   ```

2. **Install Python package**:
   ```bash
   pip install -r requirements.txt
   ```

## Key Algorithms

### Stem Modification Algorithm
1. **Split stem coordinates** into left and right halves based on base pairing
2. **Random position selection** from available positions in left half
3. **Calculate complementary position** in right half using pairing logic
4. **Insert/delete complementary bases** maintaining Watson-Crick pairing
5. **Update structure notation** with appropriate bracket symbols

### Dinucleotide Shuffling
1. **Extract dinucleotides** from original sequence
2. **Shuffle dinucleotide order** while preserving composition
3. **Reconstruct sequence** from shuffled dinucleotides
4. **Fold shuffled sequence** to obtain new structure

### Modification Normalization
- **Length-based scaling**: `factor = max(1.0, seq_length / normalization_len)`
- **Applied to all modification counts** before sampling
- **Ensures consistent modification density** across different sequence lengths

## Error Handling

The package includes comprehensive error handling for:
- Invalid sequence characters
- Structure folding failures
- Modification constraint violations
- File I/O errors
- Parallel processing exceptions

## Performance Considerations

- **Parallel processing**: Use `--num_workers > 1` for large datasets
- **Memory usage**: Adjust `--batch_size` based on available RAM
- **ViennaRNA dependency**: Structure folding is the computational bottleneck
- **Disk I/O**: Large datasets may require significant storage space

## Debugging and Logging

- **Debug mode**: `--debug` flag enables detailed logging
- **Progress tracking**: tqdm progress bars for long-running operations
- **Modification tracking**: Detailed logs of all applied modifications
- **Error reporting**: Comprehensive error messages with context

This package provides a complete toolkit for generating RNA structural datasets suitable for machine learning applications, with careful attention to biological realism and computational efficiency.