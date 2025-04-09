# Text Document Preprocessing Pipeline

This project implements a comprehensive preprocessing pipeline for text documents, specifically designed to prepare data for various locality-sensitive hashing (LSH) methods including MinHash, SimHash, and Bit Sampling. The pipeline handles data downloading, preprocessing, and storage with a focus on efficiency and scalability.

## Table of Contents
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Usage](#basic-usage)
  - [Running the Pipeline](#running-the-pipeline)
  - [Processing Wiki40b Dataset](#processing-wiki40b-dataset)
  - [Loading Processed Data](#loading-processed-data)
- [Data Preprocessing Details](#data-preprocessing-details)
- [Configuration Options](#configuration-options)
- [Data Verification](#data-verification)
- [Error Handling](#error-handling)
- [Performance Considerations](#performance-considerations)
- [Hardware Requirements](#hardware-requirements)
- [Dependencies](#dependencies)
- [Testing the Pipeline](#testing-the-pipeline)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Multiple Preprocessing Methods**:
  - K-grams generation for MinHash
  - TF-IDF vectorization for SimHash
  - Feature hashing for Bit Sampling

- **Efficient Data Handling**:
  - Batch processing for large datasets
  - Local caching of downloaded data
  - Memory-efficient sparse matrix operations
  - Automatic handling of train/validation/test splits

- **Robust Data Management**:
  - Organized directory structure
  - Metadata tracking
  - Data verification tools
  - Progress logging

## Directory Structure

```
project_root/
├── data/                      # Base directory for all data
│   ├── raw/                  # Raw downloaded datasets
│   ├── processed/            # Processed representations
│   │   ├── minhash/         # K-grams for MinHash
│   │   ├── simhash/         # TF-IDF vectors for SimHash
│   │   └── bit_sampling/    # Hashed features for bit sampling
│   └── cache/               # Cache for Hugging Face datasets
├── preprocessor.py           # Core preprocessing implementations
├── prepare_data.py          # Data preparation pipeline
├── verify_data.py           # Data verification tools
├── test_pipeline.py         # Test script to verify pipeline functionality
└── requirements.txt         # Project dependencies
```

## Installation

1. Create a conda environment:
```bash
conda create -n hash_preprocessing python=3.10
conda activate hash_preprocessing
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note**: Always ensure that the conda environment is activated for all operations:
```bash
conda activate hash_preprocessing
```

## Usage

### Basic Usage

1. Process a dataset:
```python
from prepare_data import DataPreparer

preparer = DataPreparer(base_dir="data")
processed_data = preparer.prepare_huggingface_dataset(
    dataset_name="google/wiki40b",
    subset="en",                # Specify dataset subset if available
    text_column="text",         # Column containing the text to process
    max_samples=100,           # Optional: Limit number of samples (useful for testing)
    batch_size=50              # Optional: Process data in batches to manage memory usage
)
```

### Loading Processed Data

```python
# Load MinHash representation
minhash_data = preparer.load_processed_data(
    dataset_name="google/wiki40b",
    subset="en",
    method="minhash"
)

# Load SimHash representation
simhash_data = preparer.load_processed_data(
    dataset_name="google/wiki40b",
    subset="en",
    method="simhash"
)

# Load Bit Sampling representation
bit_sampling_data = preparer.load_processed_data(
    dataset_name="google/wiki40b",
    subset="en",
    method="bit_sampling"
)
```

### Running the Pipeline

1. Test with a small dataset first:
```bash
# Make sure to activate the conda environment first
conda activate hash_preprocessing
python prepare_data.py
```
This will:
- Test the pipeline with a small dataset
- Test with a sample of wiki40b
- Ask for confirmation before processing the full dataset

2. Verify processed data:
```bash
# Make sure to activate the conda environment first
conda activate hash_preprocessing
python verify_data.py --dir data_directory --dataset dataset_name
```

### Processing Wiki40b Dataset

The main target dataset is Google's Wiki40b English subset. To process this dataset:

```python
from prepare_data import DataPreparer

# For testing with a small sample first
preparer = DataPreparer(base_dir="wiki40b_data")
processed_data = preparer.prepare_huggingface_dataset(
    dataset_name="google/wiki40b",
    subset="en",
    text_column="text",
    max_samples=10  # Use a small sample for testing
)

# For processing the full dataset (requires more memory)
preparer = DataPreparer(base_dir="wiki40b_data")
processed_data = preparer.prepare_huggingface_dataset(
    dataset_name="google/wiki40b",
    subset="en",
    text_column="text",
    batch_size=100  # Adjust based on available memory
)
```

## Data Preprocessing Details

### 1. MinHash Preprocessing
- Generates character k-grams (default k=3)
- Handles text cleaning and normalization
- Returns sets of k-grams for each document

### 2. SimHash Preprocessing
- Uses TF-IDF vectorization
- Maintains consistent vocabulary across splits
- Returns sparse matrices for memory efficiency

### 3. Bit Sampling Preprocessing
- Uses feature hashing for dimensionality reduction
- Fixed output dimension (default 1024)
- Memory-efficient representation

## Configuration Options

### DataPreprocessor
```python
preprocessor = DataPreprocessor(
    kgram_k=3,                    # K-gram size for MinHash
    tfidf_max_features=None,      # Max features for TF-IDF
    hashing_n_features=1024       # Number of features for bit sampling
)
```

### DataPreparer
```python
preparer = DataPreparer(
    base_dir="data",              # Base directory for data storage
    random_state=42,              # Random seed for reproducibility
    cache_dir=None               # Custom cache directory for datasets
)
```

#### Important Parameters for prepare_huggingface_dataset:

| Parameter | Description | Default |
|-----------|-------------|---------|
| dataset_name | Name of the dataset on Hugging Face | Required |
| text_column | Column containing the text to process | Required |
| subset | Dataset subset name (if applicable) | None |
| max_samples | Limit the number of samples to process | None (all samples) |
| batch_size | Process data in batches of this size | 1000 |

## Data Verification

The `verify_data.py` script checks:
1. Directory structure integrity
2. Raw data availability and format
3. Processed data consistency
4. Data type correctness for each method

Usage:
```bash
# Make sure to activate the conda environment first
conda activate hash_preprocessing
python verify_data.py --dir data_directory --dataset dataset_name
```

## Error Handling

- Automatic checking for missing directories
- Validation of text column existence
- Batch processing with progress tracking
- Detailed error logging

## Performance Considerations

- Uses sparse matrices for efficient memory usage
- Implements batch processing for large datasets
- Caches downloaded data locally
- Maintains consistent feature space across splits

## Hardware Requirements

For processing the full Wiki40b dataset:
- Minimum 8GB RAM recommended
- SSD storage for faster processing
- No GPU required, but can speed up processing

## Dependencies

- numpy>=1.21.0
- scikit-learn>=1.0.2
- pandas>=1.3.0
- datasets>=2.14.0
- pyarrow>=10.0.0
- scipy>=1.8.0

## Testing the Pipeline

A test script is provided to verify that the pipeline works correctly:

```bash
# Make sure to activate the conda environment first
conda activate hash_preprocessing
python test_pipeline.py
```

This will:
1. Test processing a small dataset
2. Verify the directory structure
3. Test processing a sample of the Wiki40b dataset

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 