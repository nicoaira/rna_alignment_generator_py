# RNA Triplet Generator - Python Version

A modular Python implementation of the RNA triplet generator for creating machine learning datasets with anchor/positive/negative RNA sequence triplets.

## Features

- **Modular Design**: Clean separation of concerns with dedicated modules for different functionalities
- **RNA Sequence Generation**: Random RNA sequence generation with configurable length distributions
- **Structure Folding**: Integration with ViennaRNA's RNAfold for secondary structure prediction
- **Structural Modifications**: Apply insertions/deletions to stems, hairpin loops, internal loops, bulges, and multi-loops
- **Negative Sample Generation**: Dinucleotide shuffling to preserve sequence composition
- **Appending Events**: Optional sequence appending with linkers
- **Parallel Processing**: Multi-threaded generation for improved performance
- **Dataset Splitting**: Automatic train/validation split functionality
- **Comprehensive Logging**: Detailed logging with configurable levels

## Project Structure

```
python_version/
├── main.py                    # Main entry point
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── config/                   # Configuration modules
│   ├── __init__.py
│   └── cli_parser.py         # Command-line argument parsing
├── core/                     # Core functionality modules
│   ├── __init__.py
│   ├── models.py             # Data models and classes
│   ├── rna_generator.py      # RNA sequence generation and folding
│   ├── modification_engine.py # RNA modification logic
│   ├── negative_generator.py  # Negative sample generation
│   └── dataset_generator.py  # Main dataset orchestration
└── utils/                    # Utility modules
    ├── __init__.py
    ├── logger.py             # Logging configuration
    └── output_handler.py     # Dataset saving and analysis
```

## Installation

1. **Prerequisites**: Install ViennaRNA package for RNA folding
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install vienna-rna
   
   # On macOS with Homebrew
   brew install viennarna
   
   # Or conda
   conda install -c bioconda viennarna
   ```

2. **Python Dependencies**:
   ```bash
   cd python_version
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

Generate 100 RNA triplets with default settings:

```bash
python main.py --num_structures 100
```

### Advanced Usage

Generate a larger dataset with custom parameters:

```bash
python main.py \
    --num_structures 10000 \
    --seq_min_len 50 \
    --seq_max_len 200 \
    --seq_len_distribution norm \
    --seq_len_mean 100 \
    --seq_len_sd 20 \
    --n_stem_indels_min 1 \
    --n_stem_indels_max 3 \
    --n_hloop_indels_min 1 \
    --n_hloop_indels_max 2 \
    --mod_normalization \
    --normalization_len 100 \
    --num_workers 8 \
    --output_dir output_python \
    --split \
    --train_fraction 0.8 \
    --val_fraction 0.2 \
    --debug
```

### Example: Fraction-Based Modification Sampling

The following command generates a 10,000 structure dataset where the number of
modifications in each structural element is sampled as a fraction of the element
length. All insertions and deletions are normalized by the length of the anchor
sequence.

```bash
python main.py \
  --num_structures 10000 \
  --seq_min_len 40 \
  --seq_max_len 300 \
  --seq_len_distribution norm \
  --seq_len_mean 120 \
  --seq_len_sd 70 \
  --mod_normalization \
  --normalization_len 50 \
  --neg_len_variation 20 \
  --f_stem_indels_min 0 --f_stem_indels_max 0.4 \
  --f_stem_indels_mean 0.1 --f_stem_indels_sd 0.05 \
  --stem_min_size 2 --stem_max_size 15 \
  --same_stem_max_n_mod 5 \
  --f_hloop_indels_min 0 --f_hloop_indels_max 0.5 \
  --f_hloop_indels_mean 0.1 --f_hloop_indels_sd 0.05 \
  --hloop_min_size 3 --hloop_max_size 15 \
  --same_hloop_max_n_mod 5 \
  --f_iloop_indels_min 0 --f_iloop_indels_max 0.6 \
  --f_iloop_indels_mean 0.2 --f_iloop_indels_sd 0.05 \
  --iloop_min_size 0 --iloop_max_size 15 \
  --same_iloop_max_n_mod 5 \
  --f_mloop_indels_min 0 --f_mloop_indels_max 0.7 \
  --f_mloop_indels_mean 0.2 --f_mloop_indels_sd 0.05 \
  --mloop_min_size 0 --mloop_max_size 15 \
  --same_mloop_max_n_mod 5 \
  --f_bulge_indels_min 0 --f_bulge_indels_max 0.7 \
  --f_bulge_indels_mean 0.2 --f_bulge_indels_sd 0.05 \
  --bulge_min_size 0 --bulge_max_size 15 \
  --same_bulge_max_n_mod 5 \
  --appending_event_probability 0 \
  --output_dir mod-counting-dataset \
  --num_workers 16 \

```

### Key Parameters

- `--num_structures`: Number of triplets to generate
- `--seq_min_len/--seq_max_len`: Sequence length range
- `--seq_len_distribution`: Length distribution (unif/norm)
- `--n_*_indels_min/max`: Modification count ranges for different structural elements
- `--mod_normalization`: Scale modifications by sequence length
- `--num_workers`: Number of parallel workers
- `--split`: Enable train/validation splitting
- `--debug`: Enable debug logging

## Module Overview

### Core Modules

1. **models.py**: Data structures and enums
   - `RnaTriplet`: Complete triplet data structure
   - `BulgeGraph`: RNA secondary structure representation
   - `NodeType`: Structural element classification
   - `ModificationCounts`: Tracking applied modifications

2. **rna_generator.py**: RNA sequence operations
   - `RnaGenerator`: Random sequence generation and folding
   - `BulgeGraphParser`: Structure parsing integration

3. **modification_engine.py**: Structural modifications
   - `ModificationEngine`: Applies insertions/deletions to structural elements
   - Handles stems, hairpin loops, internal loops, bulges, and multi-loops

4. **negative_generator.py**: Negative sample creation
   - `NegativeSampleGenerator`: Dinucleotide shuffling
   - `AppendingEngine`: Optional sequence appending

5. **dataset_generator.py**: Main orchestration
   - `DatasetGenerator`: Coordinates all components for triplet generation
   - Supports both sequential and parallel processing

### Utility Modules

1. **logger.py**: Logging configuration
2. **output_handler.py**: Dataset saving and analysis
   - CSV export with metadata
   - Train/validation splitting
   - Dataset statistics

## Output Format

The generator saves a CSV file containing one row per RNA triplet. The most
important columns are:

- `triplet_id` – unique identifier for the triplet
- `anchor_seq` / `anchor_structure` – original sequence and predicted structure
- `positive_seq` / `positive_structure` – sequence after applying insertions or
  deletions
- `negative_seq` / `negative_structure` – dinucleotide shuffled sequence and its
  structure
- `total_modifications` – total number of edit operations performed on the
  anchor
- `*_modifications` – per-element modification counts for stems, hairpins,
  internal loops, bulges and multi-loops
- `total_len_*` – total nucleotide length of each structural element on the
  anchor sequence
- `*_insertions` / `*_deletions` – counts of individual insertion and deletion
  events for each element
- `f_*` columns – fractional versions of the counts, normalized by the
  respective element length or by the full anchor length

Metadata describing the parameters used for generation is stored as a JSON file
and also embedded as a comment on the first line of the CSV.

### Structure Length Calculation

Lengths for individual structural elements are derived from the bulge graph
representation produced by `forgi`. For stems, hairpins, bulges and multi loops
the length is simply `end - start + 1`. Internal loops consist of two unpaired
sections, so their length is the sum of the left and right segment sizes
(`left_end - left_start + right_end - right_start + 2`). These lengths are
recorded in the `total_len_*` columns and are used when computing the fractional
`f_*` metrics.

## Comparison with Rust Version

This Python version offers several improvements over the original Rust implementation:

1. **Modularity**: Better separation of concerns with dedicated modules
2. **Readability**: More accessible codebase for researchers
3. **Extensibility**: Easier to add new features and modifications
4. **Integration**: Simpler integration with Python-based ML pipelines
5. **Debugging**: More straightforward debugging and profiling

## Performance Notes

- Use `--num_workers > 1` for parallel processing on multi-core systems
- Enable `--debug` for detailed execution logging
- The ViennaRNA dependency is required for RNA folding functionality

## Dependencies

- **ViennaRNA**: RNA secondary structure prediction
- **NumPy/SciPy**: Numerical computations and distributions
- **tqdm**: Progress bars
- **Python 3.8+**: Modern Python features

## License

This project follows the same license as the original Rust implementation.