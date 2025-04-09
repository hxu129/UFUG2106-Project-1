#!/usr/bin/env python
"""
Demo script for MinHash representation and its usage for Jaccard similarity.

This script demonstrates:
1. How to generate MinHash representations
2. How to compute Jaccard similarity between documents
3. How to find similar documents using MinHash
"""

import numpy as np
import json
import random
from pathlib import Path
import argparse
import logging
from prepare_data import DataPreparer
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def jaccard_similarity(set1, set2):
    """Compute Jaccard similarity between two sets"""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

def main():
    """Demonstrate MinHash representation usage"""
    parser = argparse.ArgumentParser(description="Demo for MinHash representation")
    parser.add_argument("--dataset", type=str, default="google/wiki40b",
                        help="Dataset name")
    parser.add_argument("--subset", type=str, default="en",
                        help="Dataset subset")
    parser.add_argument("--base-dir", type=str, default="wiki40b_data",
                        help="Base directory for data")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "validation", "test"],
                        help="Dataset split to use")
    parser.add_argument("--format", type=str, default="json",
                        choices=["json", "pickle"],
                        help="Data format to load")
    parser.add_argument("--sample-size", type=int, default=10,
                        help="Sample size for analysis")
    args = parser.parse_args()
    
    # Load data
    preparer = DataPreparer(base_dir=args.base_dir)
    minhash_data = preparer.load_processed_data(
        dataset_name=args.dataset,
        subset=args.subset,
        method="minhash",
        splits=[args.split],
        format_type=args.format
    )
    
    if not minhash_data or args.split not in minhash_data:
        logging.error(f"No {args.split} data found. Please run prepare_data.py first.")
        return
    
    data = minhash_data[args.split]
    
    # If the data is loaded from JSON, convert lists back to sets
    if args.format == "json":
        data = [set(doc) for doc in data]
    
    # Limit sample size
    sample_size = min(args.sample_size, len(data))
    sample_data = data[:sample_size]
    
    # 1. Display information about the MinHash representations
    logging.info(f"Loaded {len(data)} documents from {args.dataset} {args.subset} {args.split}")
    logging.info(f"Each document is represented as a set of k-grams")
    
    # Show size statistics
    kgram_counts = [len(doc) for doc in sample_data]
    logging.info(f"K-gram set size statistics: min={min(kgram_counts)}, max={max(kgram_counts)}, avg={sum(kgram_counts)/len(kgram_counts):.2f}")
    
    # 2. Compute Jaccard similarity
    logging.info("\nComputing Jaccard similarities between documents:")
    similarity_matrix = np.zeros((sample_size, sample_size))
    
    for i in range(sample_size):
        for j in range(sample_size):
            similarity_matrix[i, j] = jaccard_similarity(sample_data[i], sample_data[j])
    
    # Display most similar pairs
    similarity_pairs = []
    for i in range(sample_size):
        for j in range(i+1, sample_size):
            similarity_pairs.append((i, j, similarity_matrix[i, j]))
    
    similarity_pairs.sort(key=lambda x: x[2], reverse=True)
    
    logging.info("\nMost similar document pairs:")
    for i, j, sim in similarity_pairs[:5]:
        logging.info(f"Documents {i} and {j}: Jaccard similarity = {sim:.4f}")
    
    # Show some example shared k-grams
    if similarity_pairs:
        i, j, sim = similarity_pairs[0]
        intersection = sample_data[i].intersection(sample_data[j])
        logging.info(f"\nSample shared k-grams between documents {i} and {j} (similarity {sim:.4f}):")
        for kgram in list(intersection)[:10]:
            logging.info(f"  - '{kgram}'")
    
    # 3. Visualize similarity matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='viridis')
    plt.colorbar(label='Jaccard Similarity')
    plt.title(f'Document Similarity (Jaccard) - {args.dataset} {args.subset}')
    plt.xlabel('Document Index')
    plt.ylabel('Document Index')
    
    # Save the visualization
    output_file = f"minhash_similarity_{args.dataset.replace('/', '_')}_{args.subset}.png"
    plt.savefig(output_file)
    logging.info(f"\nSimilarity visualization saved to {output_file}")
    
    # 4. Demo of finding similar documents
    logging.info("\nDemo: Finding similar documents to a query document:")
    query_idx = random.randint(0, sample_size-1)
    logging.info(f"Query document index: {query_idx}")
    
    # Find similar documents using Jaccard
    similarities = [(i, jaccard_similarity(sample_data[query_idx], sample_data[i])) 
                    for i in range(sample_size) if i != query_idx]
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    logging.info("\nTop 3 similar documents:")
    for i, sim in similarities[:3]:
        logging.info(f"Document {i}: Jaccard similarity = {sim:.4f}")
    
    # Show common k-grams
    if similarities:
        i, sim = similarities[0]
        intersection = sample_data[query_idx].intersection(sample_data[i])
        logging.info(f"\nSample shared k-grams with most similar document {i} (similarity {sim:.4f}):")
        for kgram in list(intersection)[:10]:
            logging.info(f"  - '{kgram}'")

if __name__ == "__main__":
    main() 