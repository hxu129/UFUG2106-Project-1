#!/usr/bin/env python
"""
Demo script for Bit Sampling representation and its usage for document clustering.

This script demonstrates:
1. How to use Bit Sampling representations
2. How to cluster documents using the bit sampling vectors
3. How to visualize document clusters using dimensionality reduction
"""

import numpy as np
import argparse
import logging
from prepare_data import DataPreparer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Demonstrate Bit Sampling representation usage"""
    parser = argparse.ArgumentParser(description="Demo for Bit Sampling representation")
    parser.add_argument("--dataset", type=str, default="google/wiki40b",
                        help="Dataset name")
    parser.add_argument("--subset", type=str, default="en",
                        help="Dataset subset")
    parser.add_argument("--base-dir", type=str, default="wiki40b_data",
                        help="Base directory for data")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "validation", "test"],
                        help="Dataset split to use")
    parser.add_argument("--sample-size", type=int, default=100,
                        help="Sample size for analysis")
    parser.add_argument("--n-clusters", type=int, default=5,
                        help="Number of clusters for K-means")
    args = parser.parse_args()
    
    # Load data
    preparer = DataPreparer(base_dir=args.base_dir)
    bitsampling_data = preparer.load_processed_data(
        dataset_name=args.dataset,
        subset=args.subset,
        method="bit_sampling",
        splits=[args.split],
        format_type="numpy"
    )
    
    if not bitsampling_data or args.split not in bitsampling_data:
        logging.error(f"No {args.split} data found. Please run prepare_data.py first.")
        return
    
    # Get the data
    data = bitsampling_data[args.split]
    
    # Limit sample size
    sample_size = min(args.sample_size, data.shape[0])
    sample_data = data[:sample_size]
    
    # 1. Display information about the Bit Sampling representations
    logging.info(f"Loaded {data.shape[0]} documents from {args.dataset} {args.subset} {args.split}")
    logging.info(f"Each document is represented as a bit sampling vector with {data.shape[1]} features")
    
    # Show statistics
    nonzero_counts = np.count_nonzero(sample_data, axis=1)
    sparsity = 1.0 - (nonzero_counts / data.shape[1])
    logging.info(f"Bit Sampling vector statistics:")
    logging.info(f"  - Feature space dimension: {data.shape[1]}")
    logging.info(f"  - Non-zero features per doc: min={np.min(nonzero_counts)}, max={np.max(nonzero_counts)}, avg={np.mean(nonzero_counts):.2f}")
    logging.info(f"  - Sparsity: min={np.min(sparsity):.4f}, max={np.max(sparsity):.4f}, avg={np.mean(sparsity):.4f}")
    
    # Show value statistics
    logging.info(f"  - Value range: min={np.min(sample_data):.4f}, max={np.max(sample_data):.4f}")
    logging.info(f"  - Mean value: {np.mean(sample_data):.4f}")
    logging.info(f"  - Standard deviation: {np.std(sample_data):.4f}")
    
    # 2. Cluster the documents using K-means
    logging.info(f"\nClustering documents into {args.n_clusters} clusters using K-means...")
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(sample_data)
    
    # Evaluate clustering quality
    if sample_size > args.n_clusters:
        silhouette = silhouette_score(sample_data, clusters)
        calinski = calinski_harabasz_score(sample_data, clusters)
        logging.info(f"Clustering quality metrics:")
        logging.info(f"  - Silhouette score: {silhouette:.4f} (higher is better, range -1 to 1)")
        logging.info(f"  - Calinski-Harabasz index: {calinski:.4f} (higher is better)")
    
    # Count documents per cluster
    cluster_counts = np.bincount(clusters, minlength=args.n_clusters)
    for i, count in enumerate(cluster_counts):
        logging.info(f"  - Cluster {i}: {count} documents")
    
    # 3. Visualize document clusters using t-SNE
    logging.info("\nVisualizing document clusters using t-SNE dimensionality reduction...")
    
    # Reduce dimensionality for visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, sample_size-1))
    reduced_data = tsne.fit_transform(sample_data)
    
    # Plot clusters
    plt.figure(figsize=(10, 8))
    # Create a colormap with distinct colors for each cluster
    colors = plt.cm.tab10(np.linspace(0, 1, args.n_clusters))
    
    for i in range(args.n_clusters):
        # Get points in this cluster
        mask = clusters == i
        plt.scatter(reduced_data[mask, 0], reduced_data[mask, 1], 
                   s=50, c=[colors[i]], label=f'Cluster {i}')
    
    plt.title(f'Document Clusters ({args.n_clusters} clusters) - {args.dataset} {args.subset}')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.legend()
    
    # Save the visualization
    output_file = f"bitsampling_clusters_{args.dataset.replace('/', '_')}_{args.subset}.png"
    plt.savefig(output_file)
    logging.info(f"\nClustering visualization saved to {output_file}")
    
    # 4. Examine cluster centroids
    logging.info("\nExamining cluster centroids:")
    
    # Visualize cluster centroid features
    plt.figure(figsize=(12, 8))
    
    # Get the top features for each centroid (by absolute value)
    top_features_per_cluster = {}
    for i in range(args.n_clusters):
        centroid = kmeans.cluster_centers_[i]
        # Get indices of top features by absolute value
        top_indices = np.argsort(np.abs(centroid))[-10:]  # Top 10 features
        top_features_per_cluster[i] = list(zip(top_indices, centroid[top_indices]))
        
        # Log top features
        logging.info(f"Cluster {i} top features:")
        for idx, weight in top_features_per_cluster[i]:
            logging.info(f"  - Feature {idx}: {weight:.4f}")
    
    # Plot centroid comparisons
    plt.figure(figsize=(15, 10))
    
    # Plot each cluster's top features
    for i in range(min(5, args.n_clusters)):  # Show up to 5 clusters
        plt.subplot(min(5, args.n_clusters), 1, i+1)
        
        # Sort by feature weight
        features, weights = zip(*sorted(top_features_per_cluster[i], key=lambda x: x[1]))
        
        plt.barh(range(len(features)), weights)
        plt.yticks(range(len(features)), [f"F{f}" for f in features])
        plt.title(f'Cluster {i} Top Features')
        plt.tight_layout()
    
    # Save feature importance
    feature_file = f"bitsampling_features_{args.dataset.replace('/', '_')}_{args.subset}.png"
    plt.savefig(feature_file)
    logging.info(f"\nCluster feature visualization saved to {feature_file}")
    
    # 5. Document assignment example
    logging.info("\nExample document cluster assignments:")
    
    # Randomly select documents from each cluster
    for i in range(args.n_clusters):
        cluster_docs = np.where(clusters == i)[0]
        if len(cluster_docs) > 0:
            # Select a random document from this cluster
            doc_idx = np.random.choice(cluster_docs)
            
            # Get distance to centroid
            centroid = kmeans.cluster_centers_[i]
            doc_vec = sample_data[doc_idx]
            distance = np.linalg.norm(doc_vec - centroid)
            
            logging.info(f"Document {doc_idx} assigned to Cluster {i} (distance to centroid: {distance:.4f})")
            
            # Show top active features for this document
            nonzero_indices = np.nonzero(doc_vec)[0]
            if len(nonzero_indices) > 0:
                # Sort by absolute value
                sorted_indices = nonzero_indices[np.argsort(-np.abs(doc_vec[nonzero_indices]))][:5]
                logging.info(f"  Top active features: {[(idx, doc_vec[idx]) for idx in sorted_indices]}")

if __name__ == "__main__":
    main() 