#!/usr/bin/env python
"""
Demo script for SimHash representation and its usage for TF-IDF based document similarity.

This script demonstrates:
1. How to use TF-IDF SimHash representations
2. How to compute cosine similarity between documents
3. How to find similar documents using SimHash
4. How to visualize feature importance in TF-IDF vectors
"""

import numpy as np
import argparse
import logging
from prepare_data import DataPreparer
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Demonstrate SimHash representation usage"""
    parser = argparse.ArgumentParser(description="Demo for SimHash representation")
    parser.add_argument("--dataset", type=str, default="google/wiki40b",
                        help="Dataset name")
    parser.add_argument("--subset", type=str, default="en",
                        help="Dataset subset")
    parser.add_argument("--base-dir", type=str, default="wiki40b_data",
                        help="Base directory for data")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "validation", "test"],
                        help="Dataset split to use")
    parser.add_argument("--sample-size", type=int, default=10,
                        help="Sample size for analysis")
    args = parser.parse_args()
    
    # Load data
    preparer = DataPreparer(base_dir=args.base_dir)
    simhash_data = preparer.load_processed_data(
        dataset_name=args.dataset,
        subset=args.subset,
        method="simhash",
        splits=[args.split],
        format_type="numpy"
    )
    
    if not simhash_data or args.split not in simhash_data:
        logging.error(f"No {args.split} data found. Please run prepare_data.py first.")
        return
    
    # Get the data
    data = simhash_data[args.split]
    
    # Limit sample size
    sample_size = min(args.sample_size, data.shape[0])
    sample_data = data[:sample_size]
    
    # 1. Display information about the SimHash representations
    logging.info(f"Loaded {data.shape[0]} documents from {args.dataset} {args.subset} {args.split}")
    logging.info(f"Each document is represented as a TF-IDF vector with {data.shape[1]} features")
    
    # Show sparsity statistics
    nonzero_counts = np.count_nonzero(sample_data, axis=1)
    sparsity = 1.0 - (nonzero_counts / data.shape[1])
    logging.info(f"TF-IDF vector statistics:")
    logging.info(f"  - Feature space dimension: {data.shape[1]}")
    logging.info(f"  - Non-zero features per doc: min={np.min(nonzero_counts)}, max={np.max(nonzero_counts)}, avg={np.mean(nonzero_counts):.2f}")
    logging.info(f"  - Sparsity: min={np.min(sparsity):.4f}, max={np.max(sparsity):.4f}, avg={np.mean(sparsity):.4f}")
    
    # 2. Compute cosine similarity
    logging.info("\nComputing cosine similarities between documents:")
    similarity_matrix = cosine_similarity(sample_data)
    
    # Display most similar pairs
    similarity_pairs = []
    for i in range(sample_size):
        for j in range(i+1, sample_size):
            similarity_pairs.append((i, j, similarity_matrix[i, j]))
    
    similarity_pairs.sort(key=lambda x: x[2], reverse=True)
    
    logging.info("\nMost similar document pairs:")
    for i, j, sim in similarity_pairs[:5]:
        logging.info(f"Documents {i} and {j}: Cosine similarity = {sim:.4f}")
    
    # 3. Visualize similarity matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='viridis')
    plt.colorbar(label='Cosine Similarity')
    plt.title(f'Document Similarity (TF-IDF) - {args.dataset} {args.subset}')
    plt.xlabel('Document Index')
    plt.ylabel('Document Index')
    
    # Save the visualization
    output_file = f"simhash_similarity_{args.dataset.replace('/', '_')}_{args.subset}.png"
    plt.savefig(output_file)
    logging.info(f"\nSimilarity visualization saved to {output_file}")
    
    # 4. Demo of finding similar documents
    logging.info("\nDemo: Finding similar documents to a query document:")
    query_idx = random.randint(0, sample_size-1)
    logging.info(f"Query document index: {query_idx}")
    
    # Find similar documents using cosine similarity
    query_vec = sample_data[query_idx].reshape(1, -1)
    similarities = cosine_similarity(query_vec, sample_data)[0]
    similar_indices = np.argsort(-similarities)  # Sort in descending order
    
    # Skip the query document itself
    similar_indices = similar_indices[similar_indices != query_idx]
    
    logging.info("\nTop 3 similar documents:")
    for i in range(min(3, len(similar_indices))):
        idx = similar_indices[i]
        sim = similarities[idx]
        logging.info(f"Document {idx}: Cosine similarity = {sim:.4f}")
    
    # 5. Visualize feature importance
    # Get the non-zero features
    plt.figure(figsize=(12, 6))
    
    # Get top K important features by TF-IDF weight for the query document
    query_vector = sample_data[query_idx]
    nonzero_indices = np.nonzero(query_vector)[0]
    top_indices = nonzero_indices[np.argsort(-query_vector[nonzero_indices])][:20]
    
    # Plot feature weights
    plt.bar(range(len(top_indices)), query_vector[top_indices])
    plt.xlabel('Feature Index')
    plt.ylabel('TF-IDF Weight')
    plt.title(f'Top Feature Weights for Document {query_idx}')
    plt.tight_layout()
    
    # Save feature importance
    feature_file = f"simhash_features_{args.dataset.replace('/', '_')}_{args.subset}.png"
    plt.savefig(feature_file)
    logging.info(f"\nFeature importance visualization saved to {feature_file}")
    
    # 6. Compare document vectors
    if similar_indices.size > 0:
        most_similar_idx = similar_indices[0]
        
        # Plot feature comparison
        plt.figure(figsize=(12, 6))
        
        # Get shared non-zero features
        query_nonzero = set(np.nonzero(sample_data[query_idx])[0])
        similar_nonzero = set(np.nonzero(sample_data[most_similar_idx])[0])
        common_features = list(query_nonzero.intersection(similar_nonzero))
        
        if common_features:
            # Sort by query document weight
            common_features.sort(key=lambda idx: -sample_data[query_idx][idx])
            common_features = common_features[:20]  # Top 20
            
            # Plot
            width = 0.35
            x = np.arange(len(common_features))
            
            plt.bar(x - width/2, [sample_data[query_idx][idx] for idx in common_features], width, label=f'Doc {query_idx}')
            plt.bar(x + width/2, [sample_data[most_similar_idx][idx] for idx in common_features], width, label=f'Doc {most_similar_idx}')
            
            plt.xlabel('Common Feature Index')
            plt.ylabel('TF-IDF Weight')
            plt.title(f'Feature Weight Comparison between Similar Documents')
            plt.xticks(x, [f"{idx}" for idx in common_features], rotation=90)
            plt.legend()
            plt.tight_layout()
            
            # Save comparison
            comparison_file = f"simhash_comparison_{args.dataset.replace('/', '_')}_{args.subset}.png"
            plt.savefig(comparison_file)
            logging.info(f"\nFeature comparison visualization saved to {comparison_file}")

if __name__ == "__main__":
    main() 