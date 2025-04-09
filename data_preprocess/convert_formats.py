#!/usr/bin/env python
"""
Convert existing pickle files to NumPy (.npz) and JSON formats

用法:
    python convert_formats.py --dataset google/wiki40b --subset en --base-dir wiki40b_data
"""

import argparse
import numpy as np
import pickle
import json
import logging
from pathlib import Path
from prepare_data import DataPreparer, show_representations

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description='将现有的pickle文件转换为NumPy和JSON格式')
    parser.add_argument('--dataset', type=str, required=True, help='数据集名称，例如 google/wiki40b')
    parser.add_argument('--subset', type=str, default=None, help='数据集子集，例如 en')
    parser.add_argument('--base-dir', type=str, default="data", help='数据基础目录')
    parser.add_argument('--methods', type=str, nargs='+', 
                        default=['minhash', 'simhash', 'bit_sampling'],
                        help='要转换的处理方法')
    parser.add_argument('--splits', type=str, nargs='+',
                        default=['train', 'validation', 'test'],
                        help='要转换的数据分割')
    
    return parser.parse_args()

def convert_pickle_to_formats(args):
    """将pickle文件转换为NumPy和JSON格式"""
    preparer = DataPreparer(base_dir=args.base_dir)
    dataset_id = preparer._generate_dataset_id(args.dataset, args.subset)
    
    for method in args.methods:
        # 确定目录
        if method == 'minhash':
            directory = preparer.minhash_dir
        elif method == 'simhash':
            directory = preparer.simhash_dir
        elif method == 'bit_sampling':
            directory = preparer.bit_sampling_dir
        else:
            logging.warning(f"未知方法: {method}，跳过")
            continue
        
        for split in args.splits:
            pickle_path = directory / f"{dataset_id}_{split}.pkl"
            
            if not pickle_path.exists():
                logging.warning(f"找不到文件: {pickle_path}")
                continue
            
            # 加载pickle文件
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            
            logging.info(f"已加载 {method} {split} 数据: {type(data)}")
            
            # 转换为NumPy格式
            try:
                if hasattr(data, 'toarray'):
                    # 创建密集表示
                    dense_data = data.toarray()
                    # 保存为NumPy .npz文件
                    np_path = directory / f"{dataset_id}_{split}.npz"
                    np.savez_compressed(np_path, data=dense_data)
                    logging.info(f"已保存NumPy数组表示: {np_path}")
                    
                    # 保存元数据
                    meta_path = directory / f"{dataset_id}_{split}_metadata.json"
                    metadata = {
                        "shape": dense_data.shape,
                        "dtype": str(dense_data.dtype),
                        "method": method,
                        "dataset_id": dataset_id,
                        "split": split
                    }
                    with open(meta_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    logging.info(f"已保存元数据: {meta_path}")
                
                # 为MinHash转换为JSON格式
                if isinstance(data, list):
                    try:
                        # 将集合转换为列表以进行JSON序列化
                        serializable_data = [list(item) if isinstance(item, set) else item for item in data]
                        json_path = directory / f"{dataset_id}_{split}.json"
                        with open(json_path, 'w') as f:
                            json.dump(serializable_data, f)
                        logging.info(f"已保存JSON表示: {json_path}")
                    except Exception as e:
                        logging.warning(f"无法保存为JSON: {str(e)}")
            
            except Exception as e:
                logging.error(f"转换 {method} {split} 时出错: {str(e)}")
    
    logging.info("转换完成")

def main():
    args = parse_args()
    convert_pickle_to_formats(args)
    
    # 显示所有格式的表示
    logging.info("\n查看转换后的NumPy格式表示:")
    show_representations(args.dataset, args.subset, args.base_dir, format_type="numpy")
    
    logging.info("\n查看原始pickle格式表示:")
    show_representations(args.dataset, args.subset, args.base_dir, format_type="pickle")
    
    if 'minhash' in args.methods:
        logging.info("\n查看JSON格式表示 (仅限MinHash):")
        show_representations(args.dataset, args.subset, args.base_dir, methods=['minhash'], format_type="json")

if __name__ == "__main__":
    main() 