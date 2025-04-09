#!/usr/bin/env python3
"""
Test script to verify that the pipeline works as described in the README.
"""

import logging
import os
from pathlib import Path
import time

from prepare_data import DataPreparer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_basic_usage():
    """Test the basic usage examples from the README."""
    logging.info("Testing basic usage examples from the README")
    
    # Create a DataPreparer instance
    preparer = DataPreparer(base_dir="test_data")
    
    # Test with a small dataset
    logging.info("Processing a small dataset (rotten_tomatoes)")
    try:
        # Process a small sample dataset
        processed_data = preparer.prepare_huggingface_dataset(
            dataset_name="rotten_tomatoes",
            text_column="text",
            max_samples=5  # Use a very small sample for quick testing
        )
        logging.info("Successfully processed rotten_tomatoes dataset")
        
        # Try to load the processed data
        logging.info("Loading processed data for each method")
        
        # Load MinHash representation
        minhash_data = preparer.load_processed_data(
            dataset_name="rotten_tomatoes",
            method="minhash"
        )
        logging.info(f"Loaded MinHash data for splits: {list(minhash_data.keys())}")
        
        # Load SimHash representation
        simhash_data = preparer.load_processed_data(
            dataset_name="rotten_tomatoes",
            method="simhash"
        )
        logging.info(f"Loaded SimHash data for splits: {list(simhash_data.keys())}")
        
        # Load Bit Sampling representation
        bit_sampling_data = preparer.load_processed_data(
            dataset_name="rotten_tomatoes",
            method="bit_sampling"
        )
        logging.info(f"Loaded Bit Sampling data for splits: {list(bit_sampling_data.keys())}")
        
        return True
    except Exception as e:
        logging.error(f"Error testing basic usage: {str(e)}")
        return False

def test_wiki40b_sample():
    """Test processing a small sample of the wiki40b dataset."""
    logging.info("Testing with a small sample of wiki40b")
    
    # Create a DataPreparer instance
    preparer = DataPreparer(base_dir="wiki40b_data")
    
    try:
        # Process a small sample
        processed_data = preparer.prepare_huggingface_dataset(
            dataset_name="google/wiki40b",
            subset="en",
            text_column="text",
            max_samples=2  # Use a very small sample for quick testing
        )
        logging.info("Successfully processed small wiki40b sample")
        return True
    except Exception as e:
        logging.error(f"Error testing wiki40b sample: {str(e)}")
        return False

def verify_directory_structure():
    """Verify that the directory structure matches what's in the README."""
    logging.info("Verifying directory structure")
    
    # Check test_data directory structure
    base_dir = Path("test_data")
    if not base_dir.exists():
        logging.error("test_data directory does not exist")
        return False
    
    expected_dirs = [
        "raw",
        "processed",
        "processed/minhash",
        "processed/simhash", 
        "processed/bit_sampling",
        "cache"
    ]
    
    for dir_name in expected_dirs:
        dir_path = base_dir / dir_name
        if not dir_path.exists():
            logging.error(f"Directory {dir_path} does not exist")
            return False
        logging.info(f"Directory {dir_path} exists as expected")
    
    return True

def main():
    """Run tests to verify the pipeline works as described in the README."""
    start_time = time.time()
    logging.info("Starting pipeline validation")
    
    # Test basic usage
    if test_basic_usage():
        logging.info("Basic usage test passed")
        
        # Verify directory structure
        if verify_directory_structure():
            logging.info("Directory structure verification passed")
            
            # Test with wiki40b sample
            if test_wiki40b_sample():
                logging.info("Wiki40b sample test passed")
                logging.info("All tests passed successfully!")
            else:
                logging.error("Wiki40b sample test failed")
        else:
            logging.error("Directory structure verification failed")
    else:
        logging.error("Basic usage test failed")
    
    logging.info(f"Validation completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main() 