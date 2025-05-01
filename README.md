Official Code for S&P 25 paper ["Alleviating the Fear of Losing Alignment in LLM Fine-tuning"](https://arxiv.org/abs/2504.09757)

# LLM Alignment Framework

A framework for evaluating and improving language model alignment through fine-tuning and parameter recovery techniques.

## Overview

This project provides tools for:
- Fine-tuning language models with alignment objectives
- Recovering model parameters after harmful fine-tuning
- Evaluating model performance and safety
- Supporting multiple LLM architectures (Llama2, Gemma, Mistral, Qwen)

## Core Components

### 1. Fine-tuning (`run_finetune_exp.py`)
- Supports LoRA-based fine-tuning
- Handles both benign and harmful training data
- Configurable training parameters
- Supports multiple LLM architectures

### 2. Parameter Recovery (`sgdg_rollback_final.py`)
- Implements gradient-guided parameter recovery
- Supports multi-GPU training
- Features warmup steps and rollback mechanisms
- Configurable recovery rates and thresholds

### 3. Evaluation (`run_eval_exp.py`)
- Measures model performance on various tasks
- Evaluates model safety and harmful behaviors
- Supports multiple evaluation datasets
- Tracks metrics across recovery steps

### 4. Results Analysis (`run_res.py`)
- Analyzes experimental results
- Processes metrics across different models and tasks
- Generates comparative analysis

## Supported Models
- Llama2 (7B, 13B)
- Gemma 2B
- Mistral v2 7B
- Qwen 7B
- You can add more.

## Tasks
- SQL
- Cheat detection
- NL2Bash conversion
- Text summarization
- Toxicity detection


## Installation

1. Clone the repository:
```bash
git clone https://github.com/kangyangWHU/LLMAlignment.git
cd LLMAlignment
```

2. Install dependencies:
```bash
conda create -n myenv python=3.9

# Step 2: Activate the environment
conda activate myenv

# install pytorch
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121

# Step 3: Install requirements via pip
pip install -r requirements.txt
```
If you use Gemma serires, you need also install FlashAttention, which requires cuda > 12.0.

```bash
pip install flash-attn==2.6.1
```

## Usage

### 1. Fine-tuning a Model

```python
python run_finetune_exp.py
```

### 2. Running Parameter Recovery

```bash
python run_recover_exp.py
```

### 3. Evaluating Results

```bash
python run_eval_exp.py
```

### 4. Analyzing Results

```bash
python run_res.py
```

### Project Structure
```
LLMAlignment/
├── run_finetune_exp.py    # Fine-tuning experiments
├── sgdg_rollback_final.py # Parameter recovery implementation
├── run_recover_exp.py     # run parameter recovery experiments
├── run_eval_exp.py        # Evaluation pipeline
├── run_res.py             # Results analysis
├── utils/                 # Utility functions
│   ├── constant.py        # Constants and mappings
│   ├── inference_utils.py # Inference helpers
│   ├── lora_utils.py      # LoRA utilities
│   └── res_utils.py       # Results processing
├── dataset/               # datasets
└── cfg/                   # Configuration files
```


## Key Features

1. **Multi-GPU Support**
   - Distributed training and evaluation
   - Efficient parameter recovery across multiple GPUs

2. **Flexible Evaluation**
   - Support for multiple tasks
   - Customizable evaluation metrics
   - Safety evaluation

3. **Parameter Recovery**
   - Gradient-guided recovery
   - Configurable recovery strategies
   - Progress tracking and checkpointing

4. **Modular Design**
   - Easy to extend to new models
   - Configurable components
   - Reusable utilities

## Citation

If you use this code in your research, please cite:

```bibtex
@INPROCEEDINGS {,
author = { Yang, Kang and Tao, Guanhong and Chen, Xun and Xu, Jun },
booktitle = { 2025 IEEE Symposium on Security and Privacy (SP) },
title = {{ Alleviating the Fear of Losing Alignment in LLM Fine-tuning }},
year = {2025},
volume = {},
ISSN = {2375-1207},
pages = {2004-2022},
keywords = {},
doi = {10.1109/SP61157.2025.00171},
url = {https://doi.ieeecomputersociety.org/10.1109/SP61157.2025.00171},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month =May}

```
