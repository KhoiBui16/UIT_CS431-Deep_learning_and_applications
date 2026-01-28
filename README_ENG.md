# 🧮 Vietnamese Math Agent - Intelligent Multimodal Math Solving System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg" alt="Transformers">
  <img src="https://img.shields.io/badge/Gradio-UI-orange.svg" alt="Gradio">
</p>

<p align="center">
  <a href="README.md">🇻🇳 Phiên bản Tiếng Việt</a>
</p>

---

## 📑 Table of Contents

- [Description](#-description)
- [Dataset](#-dataset)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Installation Guide](#-installation-guide)
- [How to Run](#%EF%B8%8F-how-to-run)
- [Models on Hugging Face Hub](#-models-on-hugging-face-hub)
- [Tech Stack](#%EF%B8%8F-tech-stack)
- [Authors](#-authors)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 📖 Description

**Vietnamese Math Agent** is an intelligent multimodal AI system for solving mathematical problems, developed as part of the **CS431 - Deep Learning and Applications** course at **University of Information Technology - VNU-HCM (UIT)**.

The system combines:
- 🖼️ **Vision Module (Vintern-1B)**: Extract mathematical content from images (Vietnamese OCR)
- 🧠 **Math Agent (Qwen2.5-Math)**: Reasoning and solving with tool-calling capabilities (ReAct Loop)
- 🛠️ **Tool-Use Architecture**: Integrated specialized computational tools

---

## 📝 Dataset

| Information | Details |
|-------------|---------|
| **Dataset Name** | Vietnamese-395k-meta-math-MetaMathQA-gg-translated |
| **Link** | [🤗 Hugging Face](https://huggingface.co/datasets/5CD-AI/Vietnamese-395k-meta-math-MetaMathQA-gg-translated) |
| **Size** | ~395,000 samples |
| **Language** | Vietnamese |
| **Description** | MetaMathQA math dataset translated to Vietnamese, containing question-answer pairs with detailed solutions |

---

## ✨ Key Features

### 1. 👁️ Image-based Problem Recognition (Vision)
- Uses **Vintern-1B-v3.5** model (5CD-AI) - powerful Vietnamese OCR
- Accurate text extraction from math problem images
- Dynamic image processing support

### 2. 🤖 Intelligent Math Solving Agent
- **ReAct (Reasoning + Acting)** architecture: Step-by-step reasoning with tool calls
- Multiple model support:
  - `Qwen/Qwen2.5-Math-1.5B-Instruct` (Base)
  - `Qwen/Qwen2.5-Math-7B-Instruct` (Large - with 4-bit Quantization)
  - `piikerpham/Vietnamese-Qwen2.5-math-1.5B` (Vietnamese Fine-tuned)
  - Custom Fine-tuned Checkpoint

### 3. 🛠️ Computational Tools
| Tool | Description |
|------|-------------|
| **Calculator** (`evaluate`) | Expression evaluation (sin, cos, sqrt, log...) |
| **Equation Solver** (`solve_equation`) | Algebraic equation solving |
| **Unit Converter** (`convert_units`) | Unit conversion |
| **Wikipedia** (`WikipediaRetriever`) | Vietnamese Wikipedia knowledge retrieval |

### 4. 🎓 Fine-tuning Pipeline
- Fine-tune models on **Vietnamese-395k-MetaMathQA** dataset
- **LoRA** (Low-Rank Adaptation) support for VRAM efficiency
- Automated evaluation pipeline with Judge Model

### 5. 🎨 Gradio Web Interface
- User-friendly interface
- Direct math problem image upload
- Step-by-step reasoning visualization

---

## 📁 Project Structure

```
UIT_CS431-Deep_learning_and_applications/
├── README.md                     # Vietnamese documentation
├── README_ENG.md                 # English documentation
├── requirements.txt
│
├── src_agent/                    # 🤖 Agent Module
│   ├── app.py                    # Gradio Web Interface
│   ├── agent.py                  # ToolUseAgent Class (ReAct Loop)
│   ├── tools.py                  # Calculator, Solver, Wikipedia...
│   ├── vision.py                 # Vintern Vision Module (OCR)
│   ├── config.py                 # Training Configuration
│   ├── train.py                  # Agent Training Script
│   ├── eval.py                   # Evaluation Pipeline
│   ├── utils.py                  # Utility Functions
│   └── eval_result/              # Evaluation Logs
│
└── src_finetune/                 # 🔧 Fine-tuning Module
    ├── train.py                  # Training Script
    ├── configs.py                # Hyperparameters
    ├── loader.py                 # Data Loading & Cleaning
    ├── preprocess.py             # Tokenization Pipeline
    └── metrics.py                # Evaluation Metrics
```

---

## 🚀 Installation Guide

### 1. Clone Repository

```bash
git clone https://github.com/KhoiBui16/UIT_CS431-Deep_learning_and_applications.git
cd UIT_CS431-Deep_learning_and_applications
```

### 2. Create Virtual Environment

#### 🐧 Ubuntu / Linux / macOS

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Verify Python path
which python
```

#### 🪟 Windows (CMD)

```cmd
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate.bat

# Verify Python path
where python
```

#### 🪟 Windows (PowerShell)

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\Activate.ps1

# (If ExecutionPolicy error occurs, run this first)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Verify Python path
Get-Command python
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

> ⚠️ **PyTorch Note**: If you have an NVIDIA GPU, install PyTorch with CUDA support:
> ```bash
> # Example for CUDA 12.1
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

---

## ▶️ How to Run

### 🎨 Launch Web Interface (Demo)

```bash
cd src_agent
python app.py
```

Access the interface at: `http://localhost:7860` or the provided Gradio Share link.

### 🎓 Fine-tune Model

```bash
cd src_finetune
python train.py
```

### 📊 Evaluate Model

```bash
cd src_agent
python eval.py
```

---

## 🤗 Models on Hugging Face Hub

| Model | Link |
|-------|------|
| **Vietnamese Qwen2.5 Math 1.5B** | [🤗 piikerpham/Vietnamese-Qwen2.5-math-1.5B](https://huggingface.co/piikerpham/Vietnamese-Qwen2.5-math-1.5B) |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Deep Learning Framework** | PyTorch >= 2.1.0 |
| **LLM Framework** | Hugging Face Transformers >= 4.40.0 |
| **Vision Model** | Vintern-1B-v3.5 (5CD-AI) |
| **Math Model** | Qwen2.5-Math (1.5B / 7B) |
| **Fine-tuning** | PEFT (LoRA), Accelerate |
| **Quantization** | BitsAndBytes (4-bit, 8-bit) |
| **Symbolic Math** | SymPy |
| **Web Interface** | Gradio |
| **Data Processing** | Pandas, NumPy, Datasets |
| **Knowledge Retrieval** | Wikipedia API |

---

## 👥 Authors

| # | Name | Email | GitHub |
|---|------|-------|--------|
| 1 | **Bùi Nhật Anh Khôi** | khoib1601@gmail.com | [@KhoiBui16](https://github.com/KhoiBui16) |
| 2 | **Đinh Lê Bình An** | 23520004@gm.uit.edu.vn | [@BinhAnndapoet](https://github.com/BinhAnndapoet) |
| 3 | **Phạm Quốc Nam** | pikkerpham@gmail.com | [@PhamQuocNam](https://github.com/PhamQuocNam) |

---

## 📄 License

This project was developed for educational purposes as part of CS431 - UIT.

---

## 🙏 Acknowledgements

- [Hugging Face](https://huggingface.co/) - Transformers & Datasets
- [5CD-AI](https://huggingface.co/5CD-AI) - Vintern Vision Model & Vietnamese Math Dataset
- [Qwen Team](https://github.com/QwenLM) - Qwen2.5-Math Models
- [UIT - VNU-HCM](https://www.uit.edu.vn/) - CS431 Deep Learning Course
