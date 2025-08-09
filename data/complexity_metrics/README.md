# Text Complexity Analysis Pipeline

This repository contains a comprehensive text complexity analysis system that calculates over 100 different complexity metrics for text data. The system is designed to analyze text complexity across different simplification levels and generate correlations and visualizations.

## Overview

The main components of this pipeline are:
- **ComplexiMeter**: The core complexity analysis engine (`compleximeter.py`)
- **Data Processing**: Scripts to process and format datasets (`process_complexity_data.py`)
- **Analysis**: Correlation analysis and visualization tools (`analysis/compute_level_correlations.py`)
- **Visualization**: Plot generation for different complexity metrics

## 0. Requirements

### System Requirements
- Python 3.10+
- Sufficient disk space for NLTK data (~500MB)
- Internet connection for initial setup (optional, local fallbacks available)

### Python Dependencies
Install the required packages:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- `pandas==2.1.4` - Data manipulation
- `spacy==3.7.2` - NLP processing
- `nltk==3.8.1` - Natural language toolkit
- `textstat==0.7.3` - Text statistics
- `taaled==0.32` - Text analysis
- `cefrpy==1.0.1` - CEFR level analysis
- `networkx==3.2.1` - Graph analysis
- `torch` & `transformers` - Deep learning components
- `disco-score` - Discourse analysis

### Environment Setup

For automated setup on cluster systems, use:
```bash
bash setup_environment.sh
```

For manual setup:
```bash
# Download NLTK data
bash download_nltk_data.sh

# Set up local entity recognition (optional)
bash create_local_entity_recognition.sh
```

## 1. Data Placement

Place your dataset files in the `data/` directory. The system expects:

### Supported Formats
- **Parquet files** (recommended): `.parquet` files
- **JSON files**: `.json` files
- **CSV files**: `.csv` files

### File Structure
```
data/
├── your_dataset.parquet    # Main dataset
├── test_data.parquet       # Test split (optional)
└── validation_data.parquet # Validation split (optional)
```

## 2. Data Structure Requirements

The `compleximeter.py` system expects a **list of dictionaries** where each dictionary represents one text to analyze. Here's what each dictionary must contain:

### Required Fields:
- **`id`**: A unique identifier for the text (string or number)
- **`text`**: The actual text content to analyze (string)

### Optional Fields:
- **`level`**: Complexity level (integer, typically 0-10 where 0 is original and 1-10 are simplifications)

### Example Dataset Structure:
```python
dataset = [
    {
        'id': 'text_001',
        'text': 'This is the original text to analyze for complexity.',
        'level': 0
    },
    {
        'id': 'text_001_simplified_1', 
        'text': 'This is a simplified version of the text.',
        'level': 1
    },
    {
        'id': 'text_001_simplified_2',
        'text': 'This is a more simplified version.',
        'level': 2
    },
    {
        'id': 'conversation_001',
        'text': 'A longer text with multiple sentences that will be processed for various complexity metrics.',
        'level': 0
    }
]
```

### What ComplexiMeter Does:
The `process_dataset()` method will:
1. Iterate through each dictionary in your dataset
2. Extract the `text` field from each dictionary
3. Calculate all available complexity metrics for that text
4. Store the results with the corresponding `id` and `level` (if provided)

### Important Notes:
- **Text Content**: Must be a non-empty string
- **ID Field**: Used for tracking and matching results
- **Level Field**: If provided, will be preserved in output for correlation analysis
- **Empty Texts**: Will be skipped with a warning
- **Any Additional Fields**: Will be preserved in the output but not used for analysis

## 3. Configuration Files to Adjust

### Main Configuration
You need to create a data loading function that converts your dataset into the format that `compleximeter.py` expects. Here's how to modify `process_complexity_data.py` (which is called by `compleximeter.py`):

**Note**: The current `process_complexity_data.py` is hardcoded for a specific dataset structure. You'll need to modify it for your dataset.

#### Option 1: If your dataset already has levels
```python
def load_and_merge_datasets():
    """Load your dataset and convert it to the required format."""
    # Load your data (modify these lines to match your file)
    df = pd.read_parquet('data/your_dataset.parquet')
    # or df = pd.read_csv('data/your_dataset.csv')
    # or df = pd.read_json('data/your_dataset.json')
    
    # Convert to the format compleximeter.py expects
    dataset = []
    for idx, row in df.iterrows():
        # Extract text from your dataset structure
        text = row['your_text_column']  # Change this to your text column name
        text_id = row.get('your_id_column', f'text_{idx}')  # Change this to your ID column
        level = row.get('your_level_column', 0)  # Change this to your level column name
        
        dataset.append({
            'id': text_id,
            'text': text,
            'level': level
        })
    
    return dataset
```

#### Option 2: If your dataset needs level generation (like the current structure)
```python
def load_and_merge_datasets():
    """Load your dataset and convert it to the required format."""
    # Load your data
    df = pd.read_parquet('data/your_dataset.parquet')
    
    # Convert to the format compleximeter.py expects
    dataset = []
    for idx, row in df.iterrows():
        original_text = row['original']
        simplifications = row['simplifications']
        
        # Add original text (level 0)
        dataset.append({
            'id': f"row_{idx}_level_0",
            'text': original_text,
            'level': 0,
            'row_id': idx
        })
        
        # Add simplifications (levels 1-10)
        for sim_idx, sim_text in enumerate(simplifications):
            level = sim_idx + 1
            dataset.append({
                'id': f"row_{idx}_level_{level}",
                'text': sim_text,
                'level': level,
                'row_id': idx
            })
    
    return dataset
```

### Analysis Configuration
Edit `analysis/compute_level_correlations.py`:

```python
# Update these variables at the top of the file
WIRED = False  # Set to True for different dataset types
BASE_DIR = "your_output_directory/"
INPUT_FILE = "your_complexity_results.csv"
```

### ComplexiMeter Configuration
The main complexity analysis engine (`compleximeter.py`) can be configured by modifying:

- **Metrics selection**: Comment/uncomment specific metric methods in the `ComplexiMeter` class
- **API endpoints**: Modify ConceptNet and DBpedia API settings for external knowledge graphs
- **Processing parameters**: Adjust thresholds and processing options for different text types
- **Output format**: Modify the `process_dataset()` method to change output structure


## 4. Running the Complexity Analysis

### Step 1: Run the Main Script
```bash
# Run the main complexity analysis
python compleximeter.py
```

This will:
- Load your dataset from `data/`
- Convert it to the format that `compleximeter.py` expects using `process_complexity_data.py`
- Run all complexity metrics on each text
- Generate `complexity_results_with_levels.csv` in the current directory

**Important**: The current `process_complexity_data.py` is designed for datasets with `original` and `simplifications` fields. If your dataset has a different structure or already contains levels, you'll need to modify this file first.

### Step 2: Verify Output
Check that the output file contains:
- All your texts with their complexity metrics
- The `id` field for each text
- The `level` field (if provided in your dataset)
- All calculated metrics as columns

### Expected Output Files:
- `complexity_results_with_levels.csv` - Main results with all metrics and level information (in current directory)
- `complexity_results2.csv` - Intermediate results from ComplexiMeter processing (in current directory)

**Important**: The calls to `compleximeter.py` and the analysis scripts are **exactly the same** regardless of whether your dataset already contains levels or not. The only difference is in how you modify `process_complexity_data.py` to load your data.

## 5. Generating Correlations and Plots

### Step 1: Run Correlation Analysis
```bash
cd analysis
python compute_level_correlations.py
```

### Step 2: Review Generated Files
The analysis will create:

#### Correlation Results:
- `level_correlations.csv` - All levels (0-10)
- `level_correlations_no_zero.csv` - Excluding level 0
- `level_correlations_split_groups.csv` - Split analysis (1-5 vs 6-10)

#### Visualization Files:
- `plots/` directory with:
  - Individual metric plots
  - Boxplots for each metric
  - Grouped visualizations
  - Trimmed versions (outlier removal)

### Step 3: Customize Analysis
Modify `analysis/compute_level_correlations.py` to:
- Change correlation methods (Pearson, Spearman, etc.)
- Adjust plot styles and colors
- Add new visualization types
- Filter specific metrics for analysis

## 6. Advanced Usage

### Batch Processing
For large datasets, use the chunked version:
```bash
python compleximeter_chunked.py
```

### Cluster Processing
Use the SLURM scripts for cluster environments:
```bash
sbatch run_on_euler.slurm
# or
sbatch run_on_euler_optimized.slurm
```




## 7. Output Interpretation

### Correlation Results
- **Pearson correlation**: Linear relationships between metrics and complexity levels
- **Spearman correlation**: Rank-based relationships between metrics and complexity levels
- **P-values**: Statistical significance of correlations
- **Sample sizes**: Number of valid data points for each metric


### Visualization Types
- **Individual plots**: Each metric vs. complexity level
- **Boxplots**: Distribution of metric values by level
- **Grouped plots**: Multiple metrics comparison
- **Trimmed plots**: Outlier-removed versions


