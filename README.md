# 训练一个小模型

本项目是一个完整的 GPT 小语言模型实现，旨在为用户提供一个可训练的语言生成模型。该模型基于 GPT-2 架构，使用 PyTorch 框架构建，支持多种配置和训练选项。

## 特性

- **易于使用**：提供简单的配置文件，用户可以轻松修改训练参数。
- **多种数据集支持**：支持从不同数据集中训练模型，包括 Shakespeare 和 OpenWebText。
- **灵活的模型配置**：用户可以根据需求自定义模型的层数、头数和嵌入维度等参数。
- **支持分布式训练**：可以在多个 GPU 上进行分布式数据并行训练。

## 安装

确保您已安装 Python 3.7 及以上版本，并且安装了以下依赖项：

```bash
pip install torch torchvision torchaudio
pip install tiktoken requests numpy
```

## 使用方法

1. **准备数据集**：使用提供的数据准备脚本下载并准备数据集。

   ```bash
   python data/shakespeare/data_prepare.py
   ```

2. **配置训练参数**：编辑 `config/train_shakespeare_char.py` 或 `config/finetune_shakespeare.py` 文件，根据需要修改训练参数。

3. **开始训练**：运行训练脚本以开始训练模型。

   ```bash
   python train.py train_shakespeare_char.py
   ```

4. **进一步微调**：（可选）：如果需要进一步微调模型，可以使用 `finetune_shakespeare.py` 脚本。

   ```bash
   python train.py finetune_shakespeare.py
   ```

5. **生成文本**：训练完成后，使用 `sample.py` 脚本生成文本。

   ```bash
   python sample.py --out_dir=out-shakespeare-char
   ```

