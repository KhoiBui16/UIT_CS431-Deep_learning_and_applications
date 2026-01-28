# 🧮 Vietnamese Math Agent - Hệ thống Giải Toán Thông Minh Đa Phương Thức

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg" alt="Transformers">
  <img src="https://img.shields.io/badge/Gradio-UI-orange.svg" alt="Gradio">
</p>

<p align="center">
  <a href="README_ENG.md">🇬🇧 English Version</a>
</p>

---

## 📑 Mục Lục

- [Mô tả](#-mô-tả)
- [Dataset](#-dataset)
- [Tính năng chính](#-tính-năng-chính)
- [Cấu Trúc Project](#-cấu-trúc-project)
- [Hướng Dẫn Cài Đặt](#-hướng-dẫn-cài-đặt)
- [Hướng Dẫn Chạy](#%EF%B8%8F-hướng-dẫn-chạy)
- [Model trên Hugging Face Hub](#-model-trên-hugging-face-hub)
- [Tech Stack](#%EF%B8%8F-tech-stack)
- [Tác Giả](#-tác-giả)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 📖 Mô tả

**Vietnamese Math Agent** là hệ thống AI giải toán thông minh đa phương thức (multimodal) được phát triển trong khuôn khổ môn học **CS431 - Deep Learning and Applications** tại **Trường Đại học Công nghệ Thông tin - ĐHQG TP.HCM (UIT)**.

Hệ thống kết hợp:
- 🖼️ **Vision Module (Vintern-1B)**: Trích xuất nội dung toán học từ hình ảnh (OCR tiếng Việt)
- 🧠 **Math Agent (Qwen2.5-Math)**: Suy luận và giải toán với khả năng gọi công cụ (ReAct Loop)
- 🛠️ **Tool-Use Architecture**: Tích hợp các công cụ tính toán chuyên biệt

---

## 📝 Dataset

| Thông tin | Chi tiết |
|-----------|----------|
| **Tên Dataset** | Vietnamese-395k-meta-math-MetaMathQA-gg-translated |
| **Link** | [🤗 Hugging Face](https://huggingface.co/datasets/5CD-AI/Vietnamese-395k-meta-math-MetaMathQA-gg-translated) |
| **Số lượng** | ~395,000 mẫu |
| **Ngôn ngữ** | Tiếng Việt |
| **Mô tả** | Bộ dữ liệu toán học MetaMathQA được dịch sang tiếng Việt, gồm các cặp câu hỏi - câu trả lời với lời giải chi tiết |

---

## ✨ Tính năng chính

### 1. 👁️ Nhận diện Đề Bài từ Hình Ảnh (Vision)
- Sử dụng model **Vintern-1B-v3.5** (5CD-AI) - OCR tiếng Việt mạnh mẽ
- Trích xuất chính xác nội dung chữ từ ảnh bài toán
- Hỗ trợ xử lý ảnh động (Dynamic Image Processing)

### 2. 🤖 Agent Giải Toán Thông Minh
- Kiến trúc **ReAct (Reasoning + Acting)**: Suy luận từng bước và gọi công cụ
- Hỗ trợ nhiều model:
  - `Qwen/Qwen2.5-Math-1.5B-Instruct` (Base)
  - `Qwen/Qwen2.5-Math-7B-Instruct` (Large - với 4-bit Quantization)
  - `piikerpham/Vietnamese-Qwen2.5-math-1.5B` (Vietnamese Fine-tuned)
  - Custom Fine-tuned Checkpoint

### 3. 🛠️ Bộ Công Cụ Tính Toán (Tools)
| Tool | Mô tả |
|------|-------|
| **Calculator** (`evaluate`) | Tính toán biểu thức (sin, cos, sqrt, log...) |
| **Equation Solver** (`solve_equation`) | Giải phương trình đại số |
| **Unit Converter** (`convert_units`) | Chuyển đổi đơn vị đo lường |
| **Wikipedia** (`WikipediaRetriever`) | Tra cứu kiến thức Wikipedia tiếng Việt |

### 4. 🎓 Fine-tuning Pipeline
- Fine-tune model trên dataset **Vietnamese-395k-MetaMathQA**
- Hỗ trợ **LoRA** (Low-Rank Adaptation) để tiết kiệm VRAM
- Pipeline đánh giá tự động với Judge Model

### 5. 🎨 Giao Diện Web Gradio
- Giao diện thân thiện, dễ sử dụng
- Upload ảnh bài toán trực tiếp
- Hiển thị quá trình suy luận từng bước

---

## 📁 Cấu Trúc Project

```
UIT_CS431-Deep_learning_and_applications/
├── README.md
├── README_ENG.md
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

## 🚀 Hướng Dẫn Cài Đặt

### 1. Clone Repository

```bash
git clone https://github.com/KhoiBui16/UIT_CS431-Deep_learning_and_applications.git
cd UIT_CS431-Deep_learning_and_applications
```

### 2. Tạo Virtual Environment

#### 🐧 Ubuntu / Linux / macOS

```bash
# Tạo virtual environment
python3 -m venv .venv

# Kích hoạt virtual environment
source .venv/bin/activate

# Kiểm tra Python đang dùng
which python
```

#### 🪟 Windows (CMD)

```cmd
# Tạo virtual environment
python -m venv .venv

# Kích hoạt virtual environment
.venv\Scripts\activate.bat

# Kiểm tra Python đang dùng
where python
```

#### 🪟 Windows (PowerShell)

```powershell
# Tạo virtual environment
python -m venv .venv

# Kích hoạt virtual environment
.venv\Scripts\Activate.ps1

# (Nếu gặp lỗi ExecutionPolicy, chạy lệnh sau trước)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Kiểm tra Python đang dùng
Get-Command python
```

### 3. Cài Đặt Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Cài đặt các thư viện
pip install -r requirements.txt
```

> ⚠️ **Lưu ý PyTorch**: Nếu bạn có GPU NVIDIA, nên cài PyTorch với CUDA support:
> ```bash
> # Ví dụ với CUDA 12.1
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

---

## ▶️ Hướng Dẫn Chạy

### 🎨 Chạy Giao Diện Web (Demo)

```bash
cd src_agent
python app.py
```

Truy cập giao diện tại: `http://localhost:7860` hoặc link Gradio Share được cung cấp.

### 🎓 Fine-tune Model

```bash
cd src_finetune
python train.py
```

### 📊 Đánh Giá Model

```bash
cd src_agent
python eval.py
```

---

## 🤗 Model trên Hugging Face Hub

| Model | Link |
|-------|------|
| **Vietnamese Qwen2.5 Math 1.5B** | [🤗 piikerpham/Vietnamese-Qwen2.5-math-1.5B](https://huggingface.co/piikerpham/Vietnamese-Qwen2.5-math-1.5B) |

---

## 🛠️ Tech Stack

| Thành phần | Công nghệ |
|------------|-----------|
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

## 👥 Tác Giả

| STT | Họ tên | Email | GitHub |
|-----|--------|-------|--------|
| 1 | **Bùi Nhật Anh Khôi** | khoib1601@gmail.com | [@KhoiBui16](https://github.com/KhoiBui16) |
| 2 | **Đinh Lê Bình An** | 23520004@gm.uit.edu.vn | [@BinhAnndapoet](https://github.com/BinhAnndapoet) |
| 3 | **Phạm Quốc Nam** | pikkerpham@gmail.com | [@PhamQuocNam](https://github.com/PhamQuocNam) |

---

## 📄 License

Project này được phát triển cho mục đích học tập trong khuôn khổ môn CS431 - UIT.

---

## 🙏 Acknowledgements

- [Hugging Face](https://huggingface.co/) - Transformers & Datasets
- [5CD-AI](https://huggingface.co/5CD-AI) - Vintern Vision Model & Vietnamese Math Dataset
- [Qwen Team](https://github.com/QwenLM) - Qwen2.5-Math Models
- [UIT - ĐHQG TP.HCM](https://www.uit.edu.vn/) - CS431 Deep Learning Course