<div align="center">

# 🧠 BraMT

**Brain Mamba & Transformer**

</div>

BraMT 是一个 EEG 基座模型，面向 EEG 预训练与下游解码任务。通过融合 Mamba 的长序列建模能力和 Transformer 的全局交互能力，结合创新的 Criss-Cross Sequence 设计，实现更高效的 EEG 基座模型。

本仓库目前维护两个分支：

- `main`：GPU 版本。
- `npu`：华为云 NPU 版本。

## 🎓 项目背景

本项目属于浙江大学本科生科研训练（SRTP）项目，基于 [CBraMod](https://github.com/wjq-learning/CBraMod) 框架开发。

我们在 CBraMod 的 Criss-Cross Transformer 架构基础上进行创新：
- **引入 Mamba 长序列建模能力**：利用选择性扫描机制处理超长 EEG 序列
- **Criss-Cross Sequence 设计**：针对 EEG 信号的多维特性（时间-通道-频率），提出交错序列处理方案，分别沿时间和通道维度进行 Mamba 处理，再通过全局融合门实现多尺度特征交互
- **混合 Mamba-Transformer 架构**：结合局部 Mamba 的高效性和全局 Attention 的表达力，为 EEG 基座模型的预训练与下游任务微调提供统一框架

**致谢**：感谢 [CBraMod](https://github.com/wjq-learning/CBraMod) 开源项目提供的宝贵参考与启发。

## 🔍 关于项目

BraMT 主要聚焦三个目标：

1. **预训练**：在 LMDB 格式的 EEG 数据上进行掩码重建。
2. **下游微调**：适配多种 EEG 分类、二分类和回归任务。
3. **效率分析**：对序列长度、batch size、吞吐量和显存占用进行比较。

## 🏗️ 模型架构概览

<div align="center">

![BraMT Architecture Overview](assets/overview.png)

</div>

BraMT 的完整工作流程：

1. **EEG 输入** → 原始脑电信号
2. **Embedding & Masking** → 特征嵌入与掩码标记
3. **Temporal & Spatial 双路径** → 分别沿时间和通道维度处理
4. **Mamba 处理** → 每个路径独立应用 Mamba 选择性扫描
5. **Transformer 融合** → 全局上下文交互
6. **Reconstruction** → 重建被掩码的 EEG patch（预训练）或分类/回归输出（下游任务）

## 🔨 环境配置

### 1. 环境安装

BraMT 基于 Python、PyTorch、LMDB 和 Mamba 相关依赖实现。

安装 Python 依赖：

```bash
pip install -r requirements.txt
```


### 3. 推荐工作目录

仓库默认使用以下目录保存运行产物：

- `logs/`：实验日志与配置。
- `model_dir/`：保存的 checkpoint。
- `wandb/`：Weights & Biases 缓存文件。

如果要公开仓库，不建议把大 checkpoint、原始数据和运行缓存直接提交到 git。

## 🚀 预训练

预训练入口为 `pretrain_main.py`。

### 预训练目标

BraMT 在预训练阶段使用掩码重建任务：

- 默认输入形状：`[B, 19, 30, 200]`
- 默认 `need_mask=True`
- 默认 `mask_ratio=0.5`
- 损失函数：`MSELoss`

只对被 mask 的位置计算重建损失。

### 预训练数据格式

预训练数据要求为 **LMDB 格式**。
`PretrainingDataset` 会从 `dataset_dir` 中读取：

- `__keys__`
- 每个 key 对应的 patch 张量

如果你自己的 LMDB 结构一致，就可以直接用于训练。

### 预训练命令示例

```bash
python pretrain_main.py \
  --dataset_dir <path_to_lmdb_dataset> \
  --model_dir model_dir \
  --log_dir logs \
  --epochs 40 \
  --batch_size 128 \
  --lr 5e-4 \
  --weight_decay 5e-2 \
  --need_mask True \
  --mask_ratio 0.5
```

### 预训练 checkpoint

我们计划将预训练 checkpoint 上传到 **HuggingFace**。

下载后建议放到：

```text
pretrained_weights/pretrained_weights.pth
```

下游脚本可通过 `--foundation_dir` 加载该文件。

> **HuggingFace 仓库**：[SpringRainawa/BraMT](https://huggingface.co/SpringRainawa/BraMT)

## ⛵ 下游微调

下游微调入口为 `finetune_main.py`。

### 支持的下游数据集

- `FACED`
- `SEED-V`
- `PhysioNet-MI`
- `SHU-MI`
- `ISRUC`
- `CHB-MIT`
- `BCIC2020-3`
- `Mumtaz2016`
- `SEED-VIG`
- `MentalArithmetic`
- `TUEV`
- `TUAB`
- `BCIC-IV-2a`

### 任务类型对应关系

| 数据集 | 任务类型 | 说明 |
| --- | --- | --- |
| FACED | 多分类 | 情绪识别 |
| SEED-V | 多分类 | 情绪识别 |
| PhysioNet-MI | 多分类 | 运动想象 |
| SHU-MI | 二分类 | 运动想象 |
| ISRUC | 多分类 | 睡眠相关分类 |
| CHB-MIT | 二分类 | 癫痫检测 |
| BCIC2020-3 | 多分类 | EEG 解码任务 |
| Mumtaz2016 | 二分类 | 睡眠/状态分类 |
| SEED-VIG | 回归 | 警觉性预测 |
| MentalArithmetic | 二分类 | 认知负荷相关任务 |
| TUEV | 多分类 | 癫痫事件识别 |
| TUAB | 二分类 | 异常脑电检测 |
| BCIC-IV-2a | 多分类 | 运动想象 |

### 下游微调命令示例

```bash
python finetune_main.py \
  --downstream_dataset FACED \
  --datasets_dir <path_to_processed_data> \
  --model_dir models/FACED \
  --foundation_dir pretrained_weights/pretrained_weights.pth \
  --epochs 30 \
  --batch_size 32 \
  --lr 1e-5 \
  --multi_lr True \
  --axis_order True \
  --mamba_global True
```

### 微调默认设置

- `multi_lr=True`：对 Mamba、Attention 和其它参数使用分组学习率。
- `frozen=False`：默认不冻结 backbone。
- `classifier=all_patch_reps`：默认使用全部 patch 表征。

可选分类头包括：

- `all_patch_reps`
- `all_patch_reps_twolayer`
- `all_patch_reps_onelayer`
- `avgpooling_patch_reps`

## 🧠 模型设置

BraMT 由三部分组成：

### 1. PatchEmbedding

- 使用卷积提取局部 patch 模式。
- 使用 FFT 频域特征增强频谱信息。
- 使用位置编码补充时空位置信息。

### 2. HybridEncoder

- 在一个编码器中混合 Mamba block 和 Attention block。
- 默认 `stage_types = mamba,attn`。
- 默认 `depths = 6,6`。
- 支持 `axis_order=True` 的多 Mamba 序列建模。
- 支持 `mamba_global=True` 的全局上下文融合。

### 3. Projection head

- 在预训练阶段用于重建被 mask 的 EEG patch。
- 在下游任务中替换为任务特定的分类或回归头。

### 关键超参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `d_model` | 200 | 隐层维度 |
| `dim_feedforward` | 800 | 前馈网络宽度 |
| `seq_len` | 30 | patch 序列长度 |
| `nhead` | 8 | 注意力头数 |
| `stage_types` | `mamba,attn` | 混合阶段布局 |
| `depths` | `6,6` | 每个阶段的层数 |
| `axis_order` | `True` | 多 Mamba 序列顺序设置 |
| `mamba_global` | `True` | 是否启用全局 Mamba 分支 |
| `d_state` | 16 | Mamba 状态维度 |
| `d_conv` | 4 | Mamba 局部卷积核大小 |
| `expand` | 2 | Mamba 扩展倍数 |
| `conv_bias` | `True` | Mamba 卷积是否带 bias |

### 训练目标

- **预训练**：对被 mask 的 patch 做 MSE 重建。
- **多分类任务**：使用 `CrossEntropyLoss`。
- **二分类任务**：使用 `BCEWithLogitsLoss`。
- **回归任务**：使用 `MSELoss`。

## 📊 实验结果

下面结果来自结项报告中的 SQX 对比实验。

所有结果都来自 **每个数据集单独调参优化** 后的实验。

### 1. 下游性能对比

| 数据集 | 指标 | BraMT | CBraMod |
| --- | --- | --- | --- |
| SHU-MI | Acc | **0.6449 ± 0.0052** | 0.6370 ± 0.0151 |
| SHU-MI | PR-AUC | 0.6990 ± 0.0108 | **0.7139 ± 0.0088** |
| SHU-MI | ROC-AUC | **0.7057 ± 0.0065** | 0.6988 ± 0.0068 |
| FACED | Acc | **0.5567 ± 0.0052** | 0.5509 ± 0.0089 |
| FACED | Kappa | 0.4991 ± 0.0056 | **0.5041 ± 0.0122** |
| FACED | F1 | 0.5571 ± 0.0048 | **0.5618 ± 0.0093** |
| BCIC | Acc | **0.5533 ± 0.0094** | 0.5373 ± 0.0108 |
| BCIC | Kappa | **0.4417 ± 0.0118** | 0.4216 ± 0.0163 |
| BCIC | F1 | **0.5533 ± 0.0095** | 0.5383 ± 0.0096 |
| PhysioNet | Acc | 0.6012（单次） | **0.6417 ± 0.0091** |
| PhysioNet | Kappa | 0.4682（单次） | **0.5222 ± 0.0169** |
| PhysioNet | F1 | 0.6054（单次） | **0.6427 ± 0.0100** |

### 2. 结果分析

- **SHU-MI**：BraMT 在 Acc 和 ROC-AUC 上更优，但 CBraMod 在 PR-AUC 上略优。
- **FACED**：BraMT 的 Acc 更高，但 CBraMod 在 Kappa 和 F1 上略优。
- **BCIC**：BraMT 在 Acc、Kappa 和 F1 上均优于 CBraMod，说明该任务上混合结构具有优势。
- **PhysioNet**：CBraMod 在三个指标上都更强，说明该数据集上纯 Criss-Cross 风格更稳定。

总体而言，BraMT 已经在多个 EEG 任务上具备较强竞争力，尤其适合长序列和结构复杂场景。

### 3. 速度与资源分析

根据报告中的延迟 / 吞吐 / 显存对比实验，可以得到以下趋势：

- 在**短序列**场景中，Attention 的绝对延迟更低。
- 随着**序列变长**，Attention 的延迟增长最快，而 BraMT 增长更平缓。
- 在**吞吐量**方面，BraMT 通常介于纯 Attention 架构和纯 Mamba 架构之间。
- 在**显存效率**方面，随着 batch size 增大，每单位显存的吞吐量逐渐下降，而 BraMT 在大 batch 下会逐渐接近 Transformer 风格的表现。

这说明 BraMT 在长序列 EEG 建模上更稳定，也更具扩展性。

