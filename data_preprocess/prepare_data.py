import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List
import pickle
from sklearn.model_selection import train_test_split
from preprocessor import DataPreprocessor
import urllib.request
import zipfile
import logging
from pathlib import Path
from datasets import load_dataset, Dataset
import time
import hashlib
import json
import argparse
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPreparer:
    """Handles data downloading, preprocessing and storage"""
    
    def __init__(self, 
                 base_dir: str = "data",
                 random_state: int = 42,
                 cache_dir: Optional[str] = None,
                 # Representation hyperparameters
                 kgram_k: int = 3,
                 tfidf_max_features: Optional[int] = None,
                 hashing_n_features: int = 1024,
                 lowercase: bool = True,
                 remove_punctuation: bool = True):
        """Initialize the DataPreparer
        
        Args:
            base_dir: Base directory for all data
            random_state: Random seed for reproducibility
            cache_dir: Directory for Hugging Face datasets cache
            
            # Hyperparameters for different representation methods
            kgram_k: Length of character k-grams for MinHash representation (default: 3)
            tfidf_max_features: Maximum number of features for TF-IDF (SimHash) (default: None = unlimited)
            hashing_n_features: Number of features for HashingVectorizer (Bit Sampling) (default: 1024)
            lowercase: Whether to convert text to lowercase in preprocessing (default: True)
            remove_punctuation: Whether to remove punctuation in preprocessing (default: True)
        """
        # Create main directory structure
        self.base_dir = Path('./data/' + base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_base_dir = self.base_dir / "processed"
        
        # Create directories for different preprocessing methods
        self.minhash_dir = self.processed_base_dir / "minhash"
        self.simhash_dir = self.processed_base_dir / "simhash"
        self.bit_sampling_dir = self.processed_base_dir / "bit_sampling"
        
        # Cache directory for Hugging Face datasets
        self.cache_dir = cache_dir if cache_dir else str(self.base_dir / "cache")
        
        self.random_state = random_state
        
        # Initialize preprocessor with passed hyperparameters
        self.preprocessor = DataPreprocessor(
            kgram_k=kgram_k,
            tfidf_max_features=tfidf_max_features,
            hashing_n_features=hashing_n_features,
            lowercase=lowercase,
            remove_punctuation=remove_punctuation
        )
        
        # Store the hyperparameters for reference
        self.hyperparams = {
            'kgram_k': kgram_k,
            'tfidf_max_features': tfidf_max_features,
            'hashing_n_features': hashing_n_features,
            'lowercase': lowercase,
            'remove_punctuation': remove_punctuation
        }
        
        # Create directory structure
        self._create_directory_structure()
        
        # Dataset metadata
        self.metadata_file = self.base_dir / "dataset_metadata.json"
        self.metadata = self._load_metadata()
        
    def _create_directory_structure(self):
        """Create necessary directory structure"""
        dirs = [
            self.base_dir,
            self.raw_dir,
            self.processed_base_dir,
            self.minhash_dir,
            self.simhash_dir,
            self.bit_sampling_dir,
            Path(self.cache_dir)
        ]
        
        for directory in dirs:
            directory.mkdir(exist_ok=True, parents=True)
            
    def _load_metadata(self) -> Dict:
        """Load dataset metadata or create if it doesn't exist"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        else:
            metadata = {"datasets": {}}
            return metadata
    
    def _save_metadata(self):
        """Save dataset metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
            
    def _generate_dataset_id(self, dataset_name: str, subset: Optional[str] = None) -> str:
        """Generate a unique ID for the dataset
        
        Args:
            dataset_name: Name of the dataset. Should be the full name including the organization
                          e.g. 'google/wiki40b' instead of just 'wiki40b'
            subset: Dataset subset if applicable, e.g. 'en'
            
        Returns:
            A unique ID string for the dataset that's used for filenames
        """
        # If the dataset doesn't contain a slash but should be prefixed, warn and try to fix common cases
        if '/' not in dataset_name:
            # Common known mappings - add more as needed
            known_prefixes = {
                'wiki40b': 'google/wiki40b',
                'c4': 'allenai/c4',
                'squad': 'squad/squad'
            }
            
            if dataset_name in known_prefixes:
                original_name = dataset_name
                dataset_name = known_prefixes[dataset_name]
                logging.warning(f"Dataset name '{original_name}' is missing organization prefix. "
                              f"Using '{dataset_name}' instead. Please use full dataset names in the future.")
            else:
                logging.warning(f"Dataset name '{dataset_name}' may be missing organization prefix. "
                              f"This might cause issues when loading from Hugging Face. "
                              f"Example of proper format: 'google/wiki40b'")
                
        if subset:
            base_id = f"{dataset_name}/{subset}"
        else:
            base_id = dataset_name
            
        # Create a hash to ensure uniqueness
        h = hashlib.md5(base_id.encode()).hexdigest()[:8]
        return f"{base_id.replace('/', '_')}_{h}"
        
    def load_huggingface_dataset(self, 
                                dataset_name: str, 
                                text_column: str,
                                subset: Optional[str] = None,
                                max_samples: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """Load dataset from Hugging Face, checking local cache first"""
        # Generate dataset ID
        dataset_id = self._generate_dataset_id(dataset_name, subset)
        
        # Check if we have processed this dataset before
        if dataset_id in self.metadata.get("datasets", {}):
            logging.info(f"Dataset {dataset_name} found in metadata")
            dataset_info = self.metadata["datasets"][dataset_id]
            
            # Check if raw data exists locally
            raw_splits = {}
            all_splits_exist = True
            
            for split in dataset_info.get("splits", []):
                split_path = self.raw_dir / f"{dataset_id}_{split}.parquet"
                if split_path.exists():
                    logging.info(f"Loading {split} split from local cache at {split_path}")
                    raw_splits[split] = pd.read_parquet(split_path)
                else:
                    logging.info(f"Split {split} not found in local cache")
                    all_splits_exist = False
                    break
            
            if all_splits_exist:
                logging.info(f"All splits loaded from local cache")
                return raw_splits
        
        # If not in cache or not all splits exist, load from Hugging Face
        logging.info(f"Loading dataset {dataset_name} from Hugging Face")
        
        start_time = time.time()
        try:
            # Load dataset from Hugging Face
            if subset:
                dataset = load_dataset(dataset_name, subset, cache_dir=self.cache_dir)
            else:
                dataset = load_dataset(dataset_name, cache_dir=self.cache_dir)
            
            logging.info(f"Dataset loaded in {time.time() - start_time:.2f} seconds")
            
            # Convert to pandas DataFrames
            data_splits = {}
            splits = []
            
            for split_name, split_data in dataset.items():
                logging.info(f"Processing {split_name} split with {len(split_data)} samples")
                
                # Sample if max_samples is provided
                if max_samples and len(split_data) > max_samples:
                    logging.info(f"Sampling {max_samples} examples from {len(split_data)} total")
                    split_data = split_data.select(range(max_samples))
                
                # Convert to pandas
                df = split_data.to_pandas()
                
                # Ensure the text column exists
                if text_column not in df.columns:
                    available_cols = ', '.join(df.columns)
                    logging.error(f"Text column '{text_column}' not found in dataset. Available columns: {available_cols}")
                    raise ValueError(f"Text column '{text_column}' not found. Available: {available_cols}")
                
                # Save to raw directory
                raw_path = self.raw_dir / f"{dataset_id}_{split_name}.parquet"
                df.to_parquet(raw_path)
                logging.info(f"Saved raw {split_name} data to {raw_path}")
                
                data_splits[split_name] = df
                splits.append(split_name)
            
            # Update metadata
            self.metadata["datasets"][dataset_id] = {
                "name": dataset_name,
                "subset": subset,
                "text_column": text_column,
                "splits": splits,
                "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            self._save_metadata()
            
            return data_splits
        
        except Exception as e:
            logging.error(f"Error loading dataset {dataset_name}: {str(e)}")
            raise
    
    def preprocess_and_save(self,
                           data_splits: Dict[str, pd.DataFrame],
                           dataset_id: str,
                           text_column: str,
                           batch_size: int = 1000):
        """Preprocess each split and save the results by preprocessing method"""
        logging.info("Preprocessing and saving data splits")
        
        # First, fit the vectorizers on all data to maintain consistent dimensions
        logging.info("Fitting vectorizers on all data to ensure consistent dimensions")
        all_texts = []
        for split_name, df in data_splits.items():
            all_texts.extend(df[text_column].tolist())
            
        # Fit the preprocessor on all texts
        if len(all_texts) > 0:
            logging.info(f"Fitting preprocessor on {len(all_texts)} texts")
            self.preprocessor.tfidf_processor.fit(all_texts)
        
        # Process each split
        processed_data = {}
        for split_name, df in data_splits.items():
            split_start_time = time.time()
            processed_data[split_name] = {}
            
            total_samples = len(df)
            logging.info(f"Processing {total_samples} samples from {split_name} split")
            
            # Process in batches to avoid memory issues
            for i in range(0, total_samples, batch_size):
                batch_end = min(i + batch_size, total_samples)
                logging.info(f"Processing batch {i//batch_size + 1}/{(total_samples-1)//batch_size + 1} ({i}-{batch_end})")
                
                batch_df = df.iloc[i:batch_end]
                texts = batch_df[text_column].tolist()
                
                # Generate representations for this batch
                batch_minhash = self.preprocessor.prepare_for_minhash(texts)
                batch_simhash = self.preprocessor.prepare_for_simhash(texts)
                batch_bit_sampling = self.preprocessor.prepare_for_bit_sampling(texts)
                
                # Initialize the dictionaries for the first batch
                if i == 0:
                    processed_data[split_name]['minhash'] = batch_minhash
                    processed_data[split_name]['simhash'] = batch_simhash
                    processed_data[split_name]['bit_sampling'] = batch_bit_sampling
                    processed_data[split_name]['original_indices'] = list(range(batch_end))
                else:
                    # Append to existing data for subsequent batches
                    processed_data[split_name]['minhash'].extend(batch_minhash)
                    # For sparse matrices, use vstack
                    processed_data[split_name]['simhash'] = self._vstack_if_possible(
                        processed_data[split_name]['simhash'], batch_simhash
                    )
                    processed_data[split_name]['bit_sampling'] = self._vstack_if_possible(
                        processed_data[split_name]['bit_sampling'], batch_bit_sampling
                    )
                    processed_data[split_name]['original_indices'].extend(list(range(i, batch_end)))
            
            # Save each preprocessing method separately
            self._save_representation(
                processed_data[split_name]['minhash'],
                self.minhash_dir,
                dataset_id,
                split_name,
                'minhash'
            )
            
            self._save_representation(
                processed_data[split_name]['simhash'],
                self.simhash_dir,
                dataset_id,
                split_name,
                'simhash'
            )
            
            self._save_representation(
                processed_data[split_name]['bit_sampling'],
                self.bit_sampling_dir,
                dataset_id,
                split_name,
                'bit_sampling'
            )
            
            logging.info(f"Completed processing {split_name} split in {time.time() - split_start_time:.2f} seconds")
        
        return processed_data
    
    def _vstack_if_possible(self, matrix_a, matrix_b):
        """Stack matrices vertically if possible, otherwise return the input as is"""
        import scipy.sparse as sp
        
        # For the first batch, matrix_a might be None
        if matrix_a is None:
            return matrix_b
            
        # Check if both are sparse matrices
        if hasattr(matrix_a, 'shape') and hasattr(matrix_b, 'shape'):
            # If the feature dimensions don't match, we need special handling
            if matrix_a.shape[1] != matrix_b.shape[1]:
                logging.warning(f"Matrix dimensions don't match: {matrix_a.shape} vs {matrix_b.shape}")
                
                # Determine max feature dimension
                max_feat_dim = max(matrix_a.shape[1], matrix_b.shape[1])
                
                # For sparse matrices, we need to pad 
                if sp.issparse(matrix_a) and sp.issparse(matrix_b):
                    # Pad matrix_a if needed
                    if matrix_a.shape[1] < max_feat_dim:
                        matrix_a = sp.hstack([matrix_a, sp.csr_matrix((matrix_a.shape[0], max_feat_dim - matrix_a.shape[1]))])
                    
                    # Pad matrix_b if needed
                    if matrix_b.shape[1] < max_feat_dim:
                        matrix_b = sp.hstack([matrix_b, sp.csr_matrix((matrix_b.shape[0], max_feat_dim - matrix_b.shape[1]))])
            
            # Now stack them
            if sp.issparse(matrix_a) and sp.issparse(matrix_b):
                return sp.vstack([matrix_a, matrix_b])
        
        # For non-sparse matrices or unhandled cases, return the last batch
        # This is a fallback that might not be ideal
        return matrix_b
    
    def _save_representation(self, data, directory, dataset_id, split_name, method_name):
        """Save a specific representation to its directory"""
        # The original method for pickle files
        pickle_path = directory / f"{dataset_id}_{split_name}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(data, f)
        logging.info(f"Saved {method_name} raw representation for {split_name} to {pickle_path}")
        
        # Add new formats for better interoperability with ML tools
        try:
            # For sparse matrices (simhash and bit_sampling), convert to dense numpy arrays
            if hasattr(data, 'toarray'):
                # Create a dense representation
                dense_data = data.toarray()
                # Save as NumPy .npz file
                np_path = directory / f"{dataset_id}_{split_name}.npz"
                np.savez_compressed(np_path, data=dense_data)
                logging.info(f"Saved {method_name} numpy array representation for {split_name} to {np_path}")
                
                # Save metadata about the shape, etc.
                meta_path = directory / f"{dataset_id}_{split_name}_metadata.json"
                metadata = {
                    "shape": dense_data.shape,
                    "dtype": str(dense_data.dtype),
                    "method": method_name,
                    "dataset_id": dataset_id,
                    "split": split_name
                }
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
            # For list data (like MinHash), try to save in an appropriate format
            elif isinstance(data, list):
                # For MinHash k-grams lists, save as a JSON file if possible
                try:
                    # Convert sets to lists for JSON serialization
                    serializable_data = [list(item) if isinstance(item, set) else item for item in data]
                    json_path = directory / f"{dataset_id}_{split_name}.json"
                    with open(json_path, 'w') as f:
                        json.dump(serializable_data, f)
                    logging.info(f"Saved {method_name} JSON representation for {split_name} to {json_path}")
                except Exception as e:
                    logging.warning(f"Could not save {method_name} as JSON: {str(e)}")
                    
        except Exception as e:
            logging.warning(f"Error saving {method_name} in additional formats: {str(e)}")
            logging.warning("Original data is still saved in pickle format")
    
    def prepare_huggingface_dataset(self,
                                   dataset_name: str,
                                   text_column: str,
                                   subset: Optional[str] = None,
                                   max_samples: Optional[int] = None,
                                   batch_size: int = 1000) -> Dict[str, Dict[str, Any]]:
        """Full pipeline for Hugging Face datasets: load, preprocess and save"""
        # Generate dataset ID
        dataset_id = self._generate_dataset_id(dataset_name, subset)
        
        # Load raw data from Hugging Face (already split into train/val/test)
        data_splits = self.load_huggingface_dataset(
            dataset_name=dataset_name, 
            text_column=text_column,
            subset=subset,
            max_samples=max_samples
        )
        
        # Preprocess and save by method
        processed_data = self.preprocess_and_save(
            data_splits=data_splits,
            dataset_id=dataset_id,
            text_column=text_column,
            batch_size=batch_size
        )
        
        return processed_data
    
    def load_processed_data(self, 
                           dataset_name: str, 
                           method: str,
                           splits: Optional[List[str]] = None,
                           subset: Optional[str] = None,
                           format_type: str = "pickle") -> Dict[str, Any]:
        """Load preprocessed data by method
        
        Args:
            dataset_name: Name of the dataset
            method: Preprocessing method (minhash, simhash, bit_sampling)
            splits: List of splits to load
            subset: Dataset subset if applicable
            format_type: Data format to load ("pickle", "numpy", "json")
        """
        dataset_id = self._generate_dataset_id(dataset_name, subset)
        
        # Determine which directory to use
        if method == 'minhash':
            directory = self.minhash_dir
        elif method == 'simhash':
            directory = self.simhash_dir
        elif method == 'bit_sampling':
            directory = self.bit_sampling_dir
        else:
            raise ValueError(f"Unknown method: {method}. Choose from: minhash, simhash, bit_sampling")
        
        # Determine splits to load
        if not splits:
            # Try to get splits from metadata
            if dataset_id in self.metadata.get("datasets", {}):
                splits = self.metadata["datasets"][dataset_id].get("splits", ["train", "validation", "test"])
            else:
                splits = ["train", "validation", "test"]
        
        # Load data for each split
        result = {}
        for split in splits:
            # Determine file path based on format
            if format_type == "pickle":
                file_path = directory / f"{dataset_id}_{split}.pkl"
                if file_path.exists():
                    with open(file_path, 'rb') as f:
                        result[split] = pickle.load(f)
                    logging.info(f"Loaded {method} data for {split} from {file_path}")
                else:
                    logging.warning(f"No {method} data found for {split} at {file_path}")
            
            elif format_type == "numpy":
                file_path = directory / f"{dataset_id}_{split}.npz"
                if file_path.exists():
                    loaded = np.load(file_path)
                    result[split] = loaded['data']
                    logging.info(f"Loaded {method} numpy data for {split} from {file_path}")
                else:
                    logging.warning(f"No {method} numpy data found for {split} at {file_path}")
            
            elif format_type == "json":
                file_path = directory / f"{dataset_id}_{split}.json"
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        result[split] = json.load(f)
                    logging.info(f"Loaded {method} JSON data for {split} from {file_path}")
                else:
                    logging.warning(f"No {method} JSON data found for {split} at {file_path}")
            
            else:
                raise ValueError(f"Unknown format type: {format_type}. Choose from: pickle, numpy, json")
        
        return result

def test_wiki40b():
    """Test function for processing wiki40b dataset"""
    preparer = DataPreparer(base_dir="wiki40b_data")
    
    try:
        # Use a small sample size for testing
        processed_data = preparer.prepare_huggingface_dataset(
            dataset_name="google/wiki40b",
            subset="en",  # English subset
            text_column="text",
            max_samples=10,  # Small sample for testing
            batch_size=5
        )
        logging.info("Test completed successfully!")
        return True
    except Exception as e:
        logging.error(f"Test failed: {str(e)}")
        return False

def test_small_dataset():
    """Test with a smaller dataset to verify functionality"""
    preparer = DataPreparer(base_dir="test_data")
    
    try:
        # Use a small dataset for testing
        processed_data = preparer.prepare_huggingface_dataset(
            dataset_name="rotten_tomatoes",
            text_column="text",
            max_samples=20,  # Small sample for testing
            batch_size=10
        )
        logging.info("Small dataset test completed successfully!")
        
        # Test loading the processed data in different formats
        # Original pickle format
        minhash_data = preparer.load_processed_data(
            dataset_name="rotten_tomatoes",
            method="minhash",
            format_type="pickle"
        )
        
        # NumPy format
        simhash_data = preparer.load_processed_data(
            dataset_name="rotten_tomatoes",
            method="simhash",
            format_type="numpy"
        )
        
        # JSON format (for list-based data like minhash)
        minhash_json = preparer.load_processed_data(
            dataset_name="rotten_tomatoes",
            method="minhash",
            format_type="json"
        )
        
        # Default format (pickle)
        bit_sampling_data = preparer.load_processed_data(
            dataset_name="rotten_tomatoes",
            method="bit_sampling"
        )
        
        logging.info(f"Loaded minhash data for splits: {list(minhash_data.keys())}")
        logging.info(f"Loaded simhash data (numpy) for splits: {list(simhash_data.keys())}")
        logging.info(f"Loaded minhash data (json) for splits: {list(minhash_json.keys())}")
        logging.info(f"Loaded bit_sampling data for splits: {list(bit_sampling_data.keys())}")
        
        # Print a sample of the data to verify format
        if 'train' in simhash_data:
            train_data = simhash_data['train']
            logging.info(f"Simhash train data shape: {train_data.shape}")
            logging.info(f"Simhash train data type: {type(train_data)}")
            logging.info(f"Simhash train data sample (first element): {train_data[0][:5]}...")
        
        if 'train' in bit_sampling_data:
            logging.info(f"Bit sampling data type: {type(bit_sampling_data['train'])}")
            
        if 'train' in minhash_json:
            logging.info(f"Minhash JSON sample (first element): {minhash_json['train'][0][:5]}...")
        
        return True
    except Exception as e:
        logging.error(f"Small dataset test failed: {str(e)}")
        return False

def process_wiki40b_full():
    """Process the full wiki40b dataset"""
    preparer = DataPreparer(base_dir="wiki40b_data")
    
    try:
        # Process the full dataset
        processed_data = preparer.prepare_huggingface_dataset(
            dataset_name="google/wiki40b",
            subset="en",  # English subset
            text_column="text",
            batch_size=100  # Adjust based on available memory
        )
        logging.info("Wiki40b processing completed successfully!")
        return True
    except Exception as e:
        logging.error(f"Wiki40b processing failed: {str(e)}")
        return False

def main():
    """Main entry point with command line arguments"""
    parser = argparse.ArgumentParser(description="Data preparation for LSH techniques")
    parser.add_argument("--dataset", type=str, default="rotten_tomatoes", 
                       help="Dataset name, including organization (e.g., 'google/wiki40b')")
    parser.add_argument("--subset", type=str, default=None, 
                       help="Dataset subset (e.g., 'en' for wiki40b)")
    parser.add_argument("--base-dir", type=str, default="test_data", 
                       help="Base directory for data storage")
    parser.add_argument("--text-column", type=str, default="text", 
                       help="Column containing the text data")
    parser.add_argument("--max-samples", type=int, default=20, 
                       help="Maximum number of samples to process (None for all)")
    parser.add_argument("--batch-size", type=int, default=10, 
                       help="Batch size for processing")
    
    # Hyperparameters for representations
    parser.add_argument("--kgram-k", type=int, default=3, 
                       help="Length of character k-grams for MinHash (default: 3)")
    parser.add_argument("--tfidf-max-features", type=int, default=None, 
                       help="Maximum features for TF-IDF vectorizer (default: None = unlimited)")
    parser.add_argument("--hashing-n-features", type=int, default=1024, 
                       help="Number of features for HashingVectorizer (default: 1024)")
    parser.add_argument("--no-lowercase", action="store_false", dest="lowercase", 
                       help="Do not convert text to lowercase")
    parser.add_argument("--no-remove-punctuation", action="store_false", dest="remove_punctuation", 
                       help="Do not remove punctuation")
    
    # Command mode
    parser.add_argument("command", nargs="?", choices=["prepare", "show", "test"], default="test",
                       help="Command to run (prepare, show, test)")
    
    args = parser.parse_args()
    
    # If command is to show representations
    if args.command == "show":
        if len(sys.argv) >= 5:  # Need dataset, subset, base_dir
            show_representations(args.dataset, args.subset, args.base_dir)
        else:
            print("Usage: python prepare_data.py show [dataset] [subset] [base_dir]")
        return
    
    # First test with a smaller dataset
    if args.command == "test":
        test_small_dataset()
        return
        
    # Create preparer with hyperparameters
    preparer = DataPreparer(
        base_dir=args.base_dir,
        kgram_k=args.kgram_k,
        tfidf_max_features=args.tfidf_max_features,
        hashing_n_features=args.hashing_n_features,
        lowercase=args.lowercase,
        remove_punctuation=args.remove_punctuation
    )
    
    # Process the dataset
    processed_data = preparer.prepare_huggingface_dataset(
        dataset_name=args.dataset,
        subset=args.subset,
        text_column=args.text_column,
        max_samples=args.max_samples,
        batch_size=args.batch_size
    )
    
    # Show results
    show_representations(args.dataset, args.subset, args.base_dir)

def show_representations(dataset_name, subset=None, base_dir="data", methods=None, format_type="numpy"):
    """
    Utility function to display representations generated by the preprocessing pipeline.
    
    Args:
        dataset_name: Name of the dataset (e.g., "google/wiki40b")
        subset: Dataset subset (e.g., "en" for English)
        base_dir: Base directory for data
        methods: List of methods to show (e.g., ["minhash", "simhash", "bit_sampling"])
        format_type: Type of format to load ("numpy", "pickle", "json")
    
    Returns:
        Dictionary with loaded representations
    """
    if methods is None:
        methods = ["minhash", "simhash", "bit_sampling"]
    
    preparer = DataPreparer(base_dir=base_dir)
    results = {}
    
    print(f"\n=== Representations for {dataset_name} ===")
    if subset:
        print(f"Subset: {subset}")
    
    for method in methods:
        try:
            representation = preparer.load_processed_data(
                dataset_name=dataset_name,
                subset=subset,
                method=method,
                format_type=format_type
            )
            
            results[method] = representation
            print(f"\n--- {method.upper()} Representation ---")
            
            for split_name, data in representation.items():
                if data is not None:
                    if isinstance(data, np.ndarray):
                        print(f"  {split_name}: NumPy array with shape {data.shape}, dtype {data.dtype}")
                        if len(data) > 0:
                            print(f"  Sample: First row first 5 values: {data[0][:5]}")
                    elif hasattr(data, 'shape'):  # For sparse matrices
                        print(f"  {split_name}: Sparse matrix with shape {data.shape}, {data.nnz} non-zero elements")
                        if data.shape[0] > 0:
                            sample = data[0].toarray()[0][:5]
                            print(f"  Sample: First row first 5 values: {sample}")
                    elif isinstance(data, list):
                        print(f"  {split_name}: List with {len(data)} items")
                        if len(data) > 0:
                            sample_data = data[0]
                            if isinstance(sample_data, (list, set)):
                                sample = list(sample_data)[:5] if isinstance(sample_data, set) else sample_data[:5]
                                print(f"  Sample: First 5 elements of first item: {sample}")
                            else:
                                print(f"  Sample: First item: {sample_data}")
                    else:
                        print(f"  {split_name}: Unknown data type {type(data)}")
                else:
                    print(f"  {split_name}: No data available")
        except Exception as e:
            print(f"Error loading {method} representation: {str(e)}")
    
    return results

if __name__ == "__main__":
    main() 