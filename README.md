# 🌐 Multilingual Text Summarizer

A powerful deep learning-based text summarization tool capable of generating summaries in **multiple languages**. This project leverages transformer-based models to summarize long texts effectively, making it useful for cross-lingual NLP applications.

![Model Architecture](https://github.com/AtharvNawal/Multilingual-Text-Summarizer/raw/main/assets/architecture.png)

---

## 🚀 Features

- ✨ Summarizes text in multiple languages (English, Hindi, French, etc.)
- 🧠 Powered by pretrained transformer models (e.g., mBART, mT5, BART)
- 📥 Accepts input via CLI or file
- 📤 Outputs concise, readable summaries
- 💬 Tokenizer & translation integration (for multilingual support)
- 📊 ROUGE evaluation for accuracy
- 🧪 Notebook-based training and testing setup

---

## 🛠️ Tech Stack

- Python 🐍
- HuggingFace Transformers 🤗
- PyTorch or TensorFlow (depending on model)
- Jupyter Notebook
- Google Colab (for training)

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/AtharvNawal/Multilingual-Text-Summarizer.git
cd Multilingual-Text-Summarizer

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

#Run
jupyter notebook notebooks/mT5_summarization.ipynb

