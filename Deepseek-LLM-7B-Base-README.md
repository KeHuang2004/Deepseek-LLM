# Deepseek-LLM-7B-Base

## 概述

Deepseek-LLM-7B 是一个具有 70 亿参数的最先进语言模型，在一个中英混合语料库（共计 2 万亿 tokens）上从零开始训练。我们为研究社区提供两个版本：

- **Base**：预训练模型，可作为后续微调或下游任务的基础。  
- **Chat**：SFT-Chat 变体，优化了对话应用。

更多详情，请访问官方仓库：

https://github.com/deepseek-ai/deepseek-LLM

---

## 模型介绍

这是 Deepseek-LLM-7B 的 **Base** 版本。它采用多头注意力架构，拥有 70 亿参数，支持最大 4096 tokens 的上下文长度。虽然默认限制为 4096，但可通过相关技术扩展此窗口。

---

## 下载

您可以从主要托管平台获得该模型。由于网络环境（VPN/代理）限制，通常手动下载再传输到服务器是最快的方式。

- ModelScope: https://www.modelscope.cn/models/deepseek-ai/deepseek-llm-7b-base/summary  
- Hugging Face: https://huggingface.co/deepseek-ai/deepseek-llm-7b-base  
- GitHub: https://github.com/deepseek-ai/deepseek-LLM  
- GitCode mirror: https://gitcode.com/hf_mirrors/deepseek-ai/deepseek-llm-7b-base  

---

## 依赖要求

- **PyTorch**: 版本 ≥ 2.60  
- **Transformers**（来自 Hugging Face）  
- **Accelerate**  

安装命令：

```bash
pip install torch>=2.60 transformers accelerate
```

---

## 目录结构

下载并解压后，文件夹结构应如下所示：

```plain
deepseek-llm-7b-base/
├── config.json                # Model configuration (max length, architecture)
├── pytorch_model-00001-of-00002.bin
├── pytorch_model-00002-of-00002.bin
├── pytorch_model.bin.index.json
└── tokenizer/
    ├── vocab.json
    ├── merges.txt
    └── tokenizer_config.json
```

---

## 快速推理示例

以下是使用 Transformers 的最小示例：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "path/to/deepseek-llm-7b-base"
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)

# Generate text
inputs = tokenizer("Hello, world", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 许可证与引用

请参阅仓库获取许可证详情。如果在您的工作中使用此模型，请正确引用 Deepseek-LLM 论文和相关仓库。
