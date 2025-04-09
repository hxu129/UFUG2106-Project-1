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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPreparer:
    """Handles data downloading, preprocessing and storage"""
    
    def __init__(self, 
                 base_dir: str = "data",
                 random_state: int = 42,
                 cache_dir: Optional[str] = None):
        # Create main directory structure
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_base_dir = self.base_dir / "processed"
        
        # Create directories for different preprocessing methods
        self.minhash_dir = self.processed_base_dir / "minhash"
        self.simhash_dir = self.processed_base_dir / "simhash"
        self.bit_sampling_dir = self.processed_base_dir / "bit_sampling"
        
        # Cache directory for Hugging Face datasets
        self.cache_dir = cache_dir if cache_dir else str(self.base_dir / "cache")
        
        self.random_state = random_state
        self.preprocessor = DataPreprocessor()
        
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
        """Generate a unique ID for the dataset"""
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
        output_path = directory / f"{dataset_id}_{split_name}.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        logging.info(f"Saved {method_name} representation for {split_name} to {output_path}")
    
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
                           subset: Optional[str] = None) -> Dict[str, Any]:
        """Load preprocessed data by method"""
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
            file_path = directory / f"{dataset_id}_{split}.pkl"
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    result[split] = pickle.load(f)
                logging.info(f"Loaded {method} data for {split} from {file_path}")
            else:
                logging.warning(f"No {method} data found for {split} at {file_path}")
        
        return result

def test_wiki40b():
    """Test function for processing wiki40b dataset"""
    preparer = DataPreparer(base_dir="./data/wiki40b_data")
    
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
    preparer = DataPreparer(base_dir="./data/test_data")
    
    try:
        # Use a small dataset for testing
        processed_data = preparer.prepare_huggingface_dataset(
            dataset_name="rotten_tomatoes",
            text_column="text",
            max_samples=20,  # Small sample for testing
            batch_size=10
        )
        logging.info("Small dataset test completed successfully!")
        
        # Test loading the processed data
        minhash_data = preparer.load_processed_data(
            dataset_name="rotten_tomatoes",
            method="minhash"
        )
        
        simhash_data = preparer.load_processed_data(
            dataset_name="rotten_tomatoes",
            method="simhash"
        )
        
        bit_sampling_data = preparer.load_processed_data(
            dataset_name="rotten_tomatoes",
            method="bit_sampling"
        )
        
        logging.info(f"Loaded minhash data for splits: {list(minhash_data.keys())}")
        logging.info(f"Loaded simhash data for splits: {list(simhash_data.keys())}")
        logging.info(f"Loaded bit_sampling data for splits: {list(bit_sampling_data.keys())}")
        
        return True
    except Exception as e:
        logging.error(f"Small dataset test failed: {str(e)}")
        return False

def process_wiki40b_full():
    """Process the full wiki40b dataset"""
    preparer = DataPreparer(base_dir="./data/wiki40b_data")
    
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
    # First test with a smaller dataset
    logging.info("Testing with a small dataset...")
    if test_small_dataset():
        logging.info("Small dataset test passed. Now testing wiki40b with a small sample...")
        # Test with wiki40b using a very small sample
        if test_wiki40b():
            logging.info("Wiki40b test passed.")
            # Ask for confirmation before processing the full dataset
            response = input("Do you want to process the full wiki40b dataset? (y/n): ")
            if response.lower() == 'y':
                logging.info("Processing full wiki40b dataset...")
                process_wiki40b_full()
            else:
                logging.info("Full processing skipped.")
        else:
            logging.error("Wiki40b test failed. Please check the error logs.")
    else:
        logging.error("Small dataset test failed. Please check the error logs.")

if __name__ == "__main__":
    main() 