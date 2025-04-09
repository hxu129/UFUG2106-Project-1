# 文本文档预处理管道

本项目实现了一个全面的文本文档预处理管道，专为各种局部敏感哈希（LSH）方法准备数据，包括MinHash、SimHash和Bit Sampling。该管道处理数据下载、预处理和存储，注重效率和可扩展性。

## 完整处理流程

下面是完整的数据处理流程：

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  原始文本数据   │     │   基础预处理    │     │  表示生成转换   │     │   存储与加载    │
│                 │     │                 │     │                 │     │                 │
│ Hugging Face    │ ──▶ │ 小写化          │ ──▶ │ MinHash         │ ──▶ │ pickle 格式     │
│ 数据集          │     │ 清除标点         │     │ SimHash         │     │ numpy 格式      │
│ (Wiki40b等)     │     │ 标准化空白       │     │ Bit Sampling    │     │ JSON 格式       │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
                                                                                 │
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐              │
│  数据利用与应用  │     │  数据分析与可视化 │     │  数据转换      │              │
│                 │     │                 │     │                │              │
│ 相似度计算       │ ◀── │ 查看表示结构     │ ◀── │ 格式转换       │ ◀─────────────┘
│ 文档聚类         │     │ 特征可视化       │     │ 稠密/稀疏转换  │
│ 近似最近邻       │     │ 导出CSV         │     │                │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### 快速开始指南

完整处理流程命令示例（从原始数据到可视化）：

```bash
# 步骤1: 激活环境
conda activate hash_preprocessing

# 步骤2: 处理原始数据集
python prepare_data.py

# 步骤3: 将pickle格式转换为numpy和json格式
python convert_formats.py --dataset google/wiki40b --subset en --base-dir wiki40b_data

# 步骤4: 查看生成的表示
python view_representations.py --dataset google/wiki40b --subset en --base-dir wiki40b_data --visualize
```

> **重要提示**: 数据集名称必须使用完整格式 `google/wiki40b`，而不是 `wiki40b`。在所有命令中保持一致。

## 目录结构

```
project_root/
├── data/                      # 所有数据的基础目录
│   ├── raw/                  # 下载的原始数据集
│   ├── processed/            # 处理后的表示
│   │   ├── minhash/         # 用于MinHash的K-grams
│   │   ├── simhash/         # 用于SimHash的TF-IDF向量
│   │   └── bit_sampling/    # 用于bit sampling的哈希特征
│   └── cache/               # Hugging Face数据集的缓存
├── preprocessor.py           # 核心预处理实现
├── prepare_data.py          # 数据准备管道
├── verify_data.py           # 数据验证工具
├── view_representations.py  # 查看生成的表示
├── convert_formats.py       # 格式转换工具
└── requirements.txt         # 项目依赖
```

## 安装

1. 创建conda环境:
```bash
conda create -n hash_preprocessing python=3.10
conda activate hash_preprocessing
```

2. 安装依赖:
```bash
pip install -r requirements.txt
```

**注意**: 始终确保在所有操作中激活conda环境:
```bash
conda activate hash_preprocessing
```

## 数据处理流程

整个数据处理过程从原始文本到最终表示主要包括以下步骤：

### 1. 数据获取与加载

**原始数据来源**:
- 支持从Hugging Face datasets库直接加载数据集
- 主要使用Google Wiki40b数据集的英文子集作为示例
- 也支持其他包含文本字段的Hugging Face数据集

**数据加载过程**:
```python
from prepare_data import DataPreparer

# 创建数据准备器
preparer = DataPreparer(base_dir="wiki40b_data")

# 加载数据集 - 注意：必须使用完整的数据集名称"google/wiki40b"，而不是"wiki40b"
data_splits = preparer.load_huggingface_dataset(
    dataset_name="google/wiki40b",  # 完整数据集名称，包含组织/仓库
    subset="en",                    # 数据集子集(英文)
    text_column="text",             # 包含文本的列名
    max_samples=100                 # 可选：限制样本数量
)
```

**数据集命名规则**:
- 必须使用完整的Hugging Face数据集名称，如 `"google/wiki40b"`，而不是 `"wiki40b"`
- 系统内部会生成唯一ID: `google_wiki40b_en_[哈希值]`（如 `google_wiki40b_en_85625bee`）
- 所有文件都使用这个生成的ID格式保存和加载

**分割处理**:
- 自动处理数据集的训练/验证/测试分割
- 保持分割一致性，确保特征空间相同
- 对每个分割单独生成表示

### 2. 文本预处理

在生成特定表示前，所有文本都会经过基础预处理：

**基础预处理**:
- 统一转换为小写（可配置）
- 移除标点符号（可配置）
- 标准化空白字符

**批处理机制**:
- 为处理大型数据集，实现了批处理机制
- 每次处理固定数量的文档（默认1000）
- 逐批处理并合并结果，避免内存溢出

```python
# 配置批处理大小 - 必须使用完整数据集名称"google/wiki40b"
processed_data = preparer.prepare_huggingface_dataset(
    dataset_name="google/wiki40b",  # 完整数据集名称
    subset="en",
    text_column="text",
    batch_size=500  # 每批处理500个文档
)
```

### 3. 表示生成

对预处理后的文本，根据不同算法生成三种表示：

**表示生成过程**:
- MinHash：生成字符级k-grams集合
- SimHash：生成TF-IDF稀疏向量
- Bit Sampling：生成哈希特征向量

**全流程处理**:
```python
# 完整的数据处理过程 - 使用完整数据集名称
processed_data = preparer.prepare_huggingface_dataset(
    dataset_name="google/wiki40b",  # 完整数据集名称
    subset="en",
    text_column="text",
    max_samples=None,  # 处理全部样本
    batch_size=1000    # 每批1000个文档
)

# processed_data 包含三种表示的字典:
# {
#   'train': {'minhash': [...], 'simhash': [...], 'bit_sampling': [...]},
#   'validation': {'minhash': [...], 'simhash': [...], 'bit_sampling': [...]},
#   'test': {'minhash': [...], 'simhash': [...], 'bit_sampling': [...]}
# }
```

### 4. 数据存储

生成的表示按以下方式存储：

**存储格式**:
- 原始表示以Pickle格式保存（节省空间，保留原始结构）
- 稀疏矩阵转换为NumPy数组格式（便于操作）
- MinHash表示额外保存为JSON格式（提高可读性）
- 附加元数据JSON文件记录数据形状和类型

**文件命名约定**:
- 格式：`{dataset_id}_{split}.{format}`
- 由`dataset_name`和`subset`生成唯一ID
- 例如：`google_wiki40b_en_85625bee_train.pkl`

## 完整处理示例

以下是从头到尾处理数据的完整示例：

```python
from prepare_data import DataPreparer

# 1. 创建数据准备器
preparer = DataPreparer(base_dir="wiki40b_data")

# 2. 加载和预处理数据（自动完成所有步骤）
processed_data = preparer.prepare_huggingface_dataset(
    dataset_name="google/wiki40b",  # 必须使用完整数据集名称
    subset="en",
    text_column="text", 
    max_samples=10,       # 小样本用于演示
    batch_size=5
)

# 3. 加载已处理的表示
minhash_data = preparer.load_processed_data(
    dataset_name="google/wiki40b",  # 必须和处理时使用相同的数据集名称
    subset="en",
    method="minhash",
    format_type="pickle"  # 或 "numpy"、"json"
)

# 4. 查看表示内容
print(f"训练集包含 {len(minhash_data['train'])} 个文档的MinHash表示")
```

## 命令行工作流程

以下是使用命令行工具的完整工作流程：

```bash
# 1. 激活环境
conda activate hash_preprocessing

# 2. 处理小样本数据测试
python prepare_data.py

# 3. 转换格式（从pickle转为numpy和json）
python convert_formats.py --dataset google/wiki40b --subset en --base-dir wiki40b_data

# 4. 查看生成的表示
python view_representations.py --dataset google/wiki40b --subset en --base-dir wiki40b_data --visualize

# 5. 仅查看某种特定表示
python view_representations.py --dataset google/wiki40b --subset en --base-dir wiki40b_data --method minhash

# 6. 导出为CSV格式进行外部分析
python view_representations.py --dataset google/wiki40b --subset en --base-dir wiki40b_data --save-csv
```

## 表示方法详解

### 1. MinHash表示

MinHash是一种用于估计文档相似度的技术，基于文档的集合表示。

**实现原理**:
- 将文本分解为字符级K-grams（默认k=3）
- 对每个文档生成K-grams集合
- 这些K-grams集合可用于计算Jaccard相似度
- 通过对集合进行哈希和取最小值，可生成压缩的文档签名

**技术细节**:
- 文本首先经过清洗（小写化、去标点）
- 使用字符滑动窗口生成连续的k个字符组合
- 边缘处理：文本两端填充特殊字符以覆盖边缘k-grams
- 返回的是每个文档的k-grams集合（不重复元素）

**数据格式**:
- 原始格式：Python集合列表(`List[Set[str]]`)
- 储存格式：序列化为JSON以提高可读性
- 每个文档表示为其唯一k-grams的集合

**用途**:
- 适用于快速文档相似度估计
- 可用于大规模文档去重
- 支持近似最近邻搜索

### 2. SimHash表示

SimHash是一种用于快速文档相似度比较的哈希技术，考虑了词频和词语重要性。

**实现原理**:
- 使用TF-IDF（词频-逆文档频率）向量化文本
- 保留词语的权重信息，而非仅有的存在/不存在信息
- 稀疏向量表示文档，每个维度对应词汇表中的一个词

**技术细节**:
- 使用scikit-learn的TfidfVectorizer实现
- 在所有数据上拟合词汇表，确保特征空间一致性
- 返回稀疏矩阵(CSR格式)以节省内存
- 支持限制特征数量以控制维度

**数据格式**:
- 原始格式：稀疏CSR矩阵
- 储存格式：
  - 原始CSR矩阵(pickle)
  - 转换为密集NumPy数组(.npz)便于操作
  - 元数据JSON文件记录形状和类型信息

**用途**:
- 考虑词语权重的文档相似度度量
- 支持语义相关性比较
- 适用于信息检索和文档分类

### 3. Bit Sampling表示

Bit Sampling是一种基于特征哈希的降维技术，为高维稀疏数据提供紧凑表示。

**实现原理**:
- 使用特征哈希技术将任意大小的词汇映射到固定维度
- 不需要维护完整词汇表，节省内存
- 固定输出维度（默认1024）无论输入文档集大小

**技术细节**:
- 使用scikit-learn的HashingVectorizer实现
- 应用哈希函数将词语映射到固定数量的桶中
- 处理哈希冲突，将不同词语可能映射到同一维度
- 特征值符号由哈希函数决定，可能为正或负

**数据格式**:
- 原始格式：稀疏CSR矩阵
- 储存格式：
  - 原始CSR矩阵(pickle) 
  - 转换为密集NumPy数组(.npz)
  - 元数据JSON文件

**用途**:
- 适用于大规模文本集合的高效处理
- 在内存有限情况下的降维技术
- 支持流式处理，无需预先知道完整词汇表

## 数据查看

### 查看生成的表示

我们提供了多种查看生成表示的方式：

1. **使用命令行工具**:
```bash
# 查看所有表示（默认使用NumPy格式）
python view_representations.py --dataset google/wiki40b --subset en --base-dir wiki40b_data

# 只查看某种特定表示（如MinHash）
python view_representations.py --dataset google/wiki40b --subset en --base-dir wiki40b_data --method minhash

# 创建可视化图表
python view_representations.py --dataset google/wiki40b --subset en --base-dir wiki40b_data --visualize

# 以CSV格式导出数据
python view_representations.py --dataset google/wiki40b --subset en --base-dir wiki40b_data --save-csv
```

2. **在Python代码中查看**:
```python
from prepare_data import DataPreparer

# 创建DataPreparer实例
preparer = DataPreparer(base_dir="wiki40b_data")

# 加载各种表示（可选择不同格式）
minhash_data = preparer.load_processed_data(
    dataset_name="google/wiki40b",
    subset="en",
    method="minhash",
    format_type="numpy"  # 可选择 "numpy"、"pickle" 或 "json"
)
```

3. **使用简化命令**:
```bash
python prepare_data.py show google/wiki40b en wiki40b_data
```

### 转换数据格式

可以使用转换工具在不同格式间转换:

```bash
python convert_formats.py --dataset google/wiki40b --subset en --base-dir wiki40b_data
```

## 硬件要求

处理完整Wiki40b数据集:
- 建议至少8GB内存
- SSD存储以加快处理速度
- 不需要GPU，但可加速处理 

## 配置参数

各种表示方法都有可调节的超参数，可以根据具体需求进行调整：

### 超参数配置

在命令行中，可以使用以下参数：

```bash
# 使用自定义超参数
python prepare_data.py prepare --dataset google/wiki40b --subset en --base-dir wiki40b_data \
    --kgram-k 4 \
    --tfidf-max-features 5000 \
    --hashing-n-features 2048 \
    --no-lowercase \
    --batch-size 1000
```

### 超参数解释

| 参数名 | 默认值 | 作用 | 适用方法 |
|--------|--------|------|----------|
| `kgram-k` | 3 | k-gram的长度，较大的值增加特异性但减少共享特征 | MinHash |
| `tfidf-max-features` | None | TF-IDF向量的最大特征数，限制维度 | SimHash |
| `hashing-n-features` | 1024 | 哈希特征的维度 | Bit Sampling |
| `no-lowercase` | False | 设置后文本不会转为小写（保留大小写） | 所有方法 |
| `no-remove-punctuation` | False | 设置后保留标点符号 | 所有方法 |
| `batch-size` | 变化 | 批处理大小，减小可节省内存 | 所有方法 |

### 在代码中配置超参数

也可以在Python代码中直接配置这些参数：

```python
from prepare_data import DataPreparer

# 创建自定义配置的预处理器
preparer = DataPreparer(
    base_dir="wiki40b_data",
    kgram_k=4,                     # 使用4-grams而非默认的3-grams
    tfidf_max_features=5000,       # 限制TF-IDF特征数为5000
    hashing_n_features=2048,       # 设置Bit Sampling的维度为2048
    lowercase=False,               # 不转换为小写
    remove_punctuation=False       # 不移除标点符号
)
```

### 超参数选择建议

- **MinHash (k-gram长度)**:
  - 较小的k (2-3): 更多共享特征，更高的召回率，适合短文本
  - 较大的k (4-5): 更精确，对文档语义更敏感，适合长文本
  - 权衡: 较大的k会产生更多不同的k-grams，需要更多内存

- **SimHash (TF-IDF最大特征数)**:
  - 无限制: 保留所有特征，最精确但维度可能非常高
  - 1000-5000: 在精度和效率间的良好平衡点
  - 权衡: 超参数越小处理越快，但可能损失信息

- **Bit Sampling (哈希特征数)**:
  - 512: 非常紧凑的表示，适合超大规模数据集
  - 1024-2048: 平衡表示能力和内存/计算需求
  - 4096+: 减少哈希冲突，但有更大的内存开销

## 演示脚本

为帮助理解和运用每种表示方法，我们提供了三个演示脚本:

### 1. MinHash演示

演示MinHash表示如何用于计算文档间的Jaccard相似度:

```bash
python demo_minhash.py --dataset google/wiki40b --subset en --base-dir wiki40b_data --sample-size 20
```

此脚本展示:
- 如何加载和使用MinHash表示
- 如何计算文档间的Jaccard相似度
- 如何可视化相似度矩阵
- 如何找到最相似的文档对

### 2. SimHash演示

展示如何使用TF-IDF向量(SimHash)进行文档分析和相似度计算:

```bash
python demo_simhash.py --dataset google/wiki40b --subset en --base-dir wiki40b_data --sample-size 20
```

此脚本展示:
- 如何使用TF-IDF向量进行相似度计算
- 如何分析向量特征的重要性
- 如何比较不同文档的向量表示

### 3. Bit Sampling演示

演示如何使用Bit Sampling向量进行文档聚类:

```bash
python demo_bit_sampling.py --dataset google/wiki40b --subset en --base-dir wiki40b_data --sample-size 100 --n-clusters 5
```

此脚本展示:
- 如何使用Bit Sampling向量进行文档聚类
- 如何评估聚类质量
- 如何可视化文档簇
- 如何分析聚类中心的特征

### 运行演示的前提条件

在运行这些演示前，必须先完成数据预处理:

1. 首先处理数据:
```bash
python prepare_data.py prepare --dataset google/wiki40b --subset en --base-dir wiki40b_data
```

2. 将pickle格式转换为numpy和json:
```bash
python convert_formats.py --dataset google/wiki40b --subset en --base-dir wiki40b_data
```

3. 然后才能运行演示脚本 