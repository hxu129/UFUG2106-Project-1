#!/usr/bin/env python
"""
视图工具：查看和分析数据预处理生成的表示

用法:
    python view_representations.py --dataset google/wiki40b --subset en --base-dir wiki40b_data
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys
import os
from prepare_data import DataPreparer, show_representations

def parse_args():
    parser = argparse.ArgumentParser(description='查看和分析数据表示')
    parser.add_argument('--dataset', type=str, required=True, help='数据集名称，例如 google/wiki40b')
    parser.add_argument('--subset', type=str, default=None, help='数据集子集，例如 en')
    parser.add_argument('--base-dir', type=str, default="data", help='数据基础目录')
    parser.add_argument('--method', type=str, default=None, 
                        choices=['minhash', 'simhash', 'bit_sampling', 'all'],
                        help='要查看的处理方法。默认为全部查看')
    parser.add_argument('--format', type=str, default='numpy', 
                        choices=['numpy', 'pickle', 'json'],
                        help='要加载的数据格式')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'validation', 'test', 'all'],
                        help='要查看的数据分割')
    parser.add_argument('--visualize', action='store_true', help='是否创建可视化图表')
    parser.add_argument('--save-csv', action='store_true', help='将表示保存为CSV文件')
    parser.add_argument('--sample-size', type=int, default=5, help='要显示的样本数量')
    
    return parser.parse_args()

def visualize_representation(data, method, split, dataset_name, subset=None, save_dir=None):
    """创建数据表示的可视化图表"""
    plt.figure(figsize=(12, 6))
    
    if isinstance(data, np.ndarray):
        # 对于密集数组，我们可以创建热力图
        if data.shape[0] > 10:
            # 如果样本太多，只显示前10个
            data_to_plot = data[:10]
        else:
            data_to_plot = data
            
        plt.imshow(data_to_plot, aspect='auto', cmap='viridis')
        plt.colorbar(label='Value')
        plt.title(f'{method.capitalize()} Representation - {split} split')
        plt.xlabel('Feature Index')
        plt.ylabel('Sample Index')
        
    elif hasattr(data, 'toarray'):
        # 对于稀疏矩阵
        if data.shape[0] > 10:
            data_to_plot = data[:10].toarray()
        else:
            data_to_plot = data.toarray()
            
        plt.imshow(data_to_plot, aspect='auto', cmap='viridis')
        plt.colorbar(label='Value')
        plt.title(f'{method.capitalize()} Representation (Sparse) - {split} split')
        plt.xlabel('Feature Index')
        plt.ylabel('Sample Index')
        
    elif isinstance(data, list):
        # 对于列表类型的数据（如minhash的k-grams集合）
        # 我们可以显示每个文档的唯一k-gram数量
        if all(isinstance(item, (list, set)) for item in data):
            lengths = [len(item) for item in data[:50]]  # 最多显示50个样本
            plt.bar(range(len(lengths)), lengths)
            plt.title(f'{method.capitalize()} - Number of unique elements per document')
            plt.xlabel('Document Index')
            plt.ylabel('Number of Elements')
    
    # 保存图表
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        dataset_id = f"{dataset_name.replace('/', '_')}"
        if subset:
            dataset_id += f"_{subset}"
        plt.savefig(save_dir / f"{dataset_id}_{method}_{split}.png", dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()

def save_to_csv(data, method, split, dataset_name, subset=None, save_dir=None):
    """保存表示为CSV文件，便于在其他工具中分析"""
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        dataset_id = f"{dataset_name.replace('/', '_')}"
        if subset:
            dataset_id += f"_{subset}"
            
        filename = save_dir / f"{dataset_id}_{method}_{split}.csv"
        
        # 转换数据为适合CSV的格式
        if isinstance(data, np.ndarray):
            # 对于NumPy数组
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            print(f"保存为CSV: {filename}")
            
        elif hasattr(data, 'toarray'):
            # 对于稀疏矩阵
            df = pd.DataFrame(data.toarray())
            df.to_csv(filename, index=False)
            print(f"保存为CSV: {filename}")
            
        elif isinstance(data, list):
            # 对于列表类型，需要特殊处理
            if all(isinstance(item, (list, set)) for item in data):
                # 对于集合列表，我们可以保存每个文档的元素数量
                lengths = [len(item) for item in data]
                df = pd.DataFrame({'document_index': range(len(lengths)), 'num_elements': lengths})
                df.to_csv(filename, index=False)
                print(f"保存元素计数为CSV: {filename}")
                
                # 另外保存前几个文档的实际内容
                sample_size = min(5, len(data))
                for i in range(sample_size):
                    sample_filename = save_dir / f"{dataset_id}_{method}_{split}_doc{i}.csv"
                    elements = list(data[i]) if isinstance(data[i], set) else data[i]
                    sample_df = pd.DataFrame({'element': elements})
                    sample_df.to_csv(sample_filename, index=False)
                    print(f"保存样本文档 {i} 为CSV: {sample_filename}")
            else:
                # 其他类型的列表
                try:
                    df = pd.DataFrame(data)
                    df.to_csv(filename, index=False)
                    print(f"保存为CSV: {filename}")
                except Exception as e:
                    print(f"无法保存为CSV: {str(e)}")

def main():
    args = parse_args()
    
    # 确定要查看的方法
    methods = None
    if args.method and args.method != 'all':
        methods = [args.method]
    
    # 确定要查看的分割
    splits = None
    if args.split and args.split != 'all':
        splits = [args.split]
    
    # 加载表示
    preparer = DataPreparer(base_dir=args.base_dir)
    
    # 如果指定了方法，只显示该方法
    if methods:
        for method in methods:
            representation = preparer.load_processed_data(
                dataset_name=args.dataset,
                subset=args.subset,
                method=method,
                format_type=args.format,
                splits=splits
            )
            
            # 显示数据基本信息
            print(f"\n=== {method.upper()} Representation ===")
            for split_name, data in representation.items():
                print_representation_info(data, split_name, args.sample_size)
                
                # 可视化
                if args.visualize:
                    visualize_representation(
                        data, method, split_name, args.dataset, args.subset,
                        save_dir=Path(args.base_dir) / "visualizations"
                    )
                
                # 保存为CSV
                if args.save_csv:
                    save_to_csv(
                        data, method, split_name, args.dataset, args.subset,
                        save_dir=Path(args.base_dir) / "csv_exports"
                    )
    else:
        # 使用通用显示函数显示所有表示
        results = show_representations(
            args.dataset, 
            args.subset, 
            args.base_dir, 
            format_type=args.format
        )
        
        # 如果需要可视化或保存为CSV，单独处理
        if args.visualize or args.save_csv:
            for method, representation in results.items():
                for split_name, data in representation.items():
                    if args.visualize:
                        visualize_representation(
                            data, method, split_name, args.dataset, args.subset,
                            save_dir=Path(args.base_dir) / "visualizations"
                        )
                    
                    if args.save_csv:
                        save_to_csv(
                            data, method, split_name, args.dataset, args.subset,
                            save_dir=Path(args.base_dir) / "csv_exports"
                        )

def print_representation_info(data, split_name, sample_size=5):
    """打印表示的详细信息"""
    if data is not None:
        if isinstance(data, np.ndarray):
            print(f"  {split_name}: NumPy数组，形状: {data.shape}, 类型: {data.dtype}")
            print(f"  数据统计: 平均值={np.mean(data):.4f}, 最小值={np.min(data):.4f}, 最大值={np.max(data):.4f}")
            if len(data) > 0:
                print(f"  样本: 前{min(sample_size, len(data))}个文档的前5个值:")
                for i in range(min(sample_size, len(data))):
                    print(f"    文档 {i}: {data[i][:5]}")
                
        elif hasattr(data, 'shape'):  # 对于稀疏矩阵
            print(f"  {split_name}: 稀疏矩阵，形状: {data.shape}, 非零元素数: {data.nnz}")
            print(f"  稀疏度: {data.nnz / (data.shape[0] * data.shape[1]):.6f} ({data.nnz} / {data.shape[0] * data.shape[1]})")
            if data.shape[0] > 0:
                print(f"  样本: 前{min(sample_size, data.shape[0])}个文档的前5个值:")
                for i in range(min(sample_size, data.shape[0])):
                    sample = data[i].toarray()[0][:5]
                    print(f"    文档 {i}: {sample}")
                    
        elif isinstance(data, list):
            print(f"  {split_name}: 列表，包含 {len(data)} 个项目")
            if len(data) > 0:
                sample_data = data[0]
                if isinstance(sample_data, (list, set)):
                    print(f"  元素类型: {type(sample_data).__name__}")
                    print(f"  统计: 平均元素数量={sum(len(item) for item in data)/len(data):.2f}")
                    print(f"  样本: 前{min(sample_size, len(data))}个文档的前5个元素:")
                    for i in range(min(sample_size, len(data))):
                        sample = list(data[i])[:5] if isinstance(data[i], set) else data[i][:5]
                        print(f"    文档 {i}: {sample}")
                else:
                    print(f"  元素类型: {type(sample_data).__name__}")
                    print(f"  样本: 前{min(sample_size, len(data))}个项目:")
                    for i in range(min(sample_size, len(data))):
                        print(f"    项目 {i}: {data[i]}")
        else:
            print(f"  {split_name}: 未知数据类型 {type(data)}")
    else:
        print(f"  {split_name}: 无可用数据")

if __name__ == "__main__":
    main() 