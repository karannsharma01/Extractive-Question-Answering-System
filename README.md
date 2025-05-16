
# ❓ Question Answering System using RoBERTa

This repository contains the implementation of a Question Answering (QA) system developed as Machine Learning Intern at **NIT Hamirpur**, focusing on fine-tuning transformer-based models for extractive QA tasks. The project explores state-of-the-art NLP techniques to build a system that can accurately extract answers to questions from given contexts.

---

## 🧠 Introduction

Question Answering is a key area in Natural Language Processing (NLP) that involves building systems capable of understanding natural language and returning precise answers. In this project, we fine-tuned **RoBERTa**—a robustly optimized variant of BERT—on the **SQuAD2.0 dataset** and a custom dataset based on movie reviews, formatted for span-based QA.

---

## ⚙️ Approach

We followed a detailed and modular pipeline to implement and evaluate the QA model:

### 1. **Data Preparation**
- Used both SQuAD2.0 and a custom dataset in `.csv` format containing `context`, `question`, and `answer`.
- Cleaned, preprocessed, and structured the data to be compatible with the Hugging Face Transformers format.

### 2. **Model Fine-Tuning**
- Used `deepset/roberta-base-squad2` from Hugging Face as a base model.
- Fine-tuned it on our custom dataset using PyTorch and the Hugging Face Trainer API.
- Training included:
  - Tokenization with offset mappings
  - Preprocessing of both training and validation examples
  - Hyperparameter tuning for epochs and batch size

### 3. **Evaluation Metrics**
- **F1 Score:** 68.07%
- **Exact Match (EM):** 66.83%
- **Inference Time:** 2.7 seconds per example
- Compared with other models like BERT, T5, and GPT-3 to highlight domain-specific tuning effectiveness.

---

## 📊 Results

| Model     | EM Score | F1 Score | Inference Time |
|-----------|----------|----------|----------------|
| RoBERTa   | 66.83%   | 68.07%   | 2.7s           |
| BERT      | 80.0%    | 85.9%    | 0.9s           |
| T5        | 81.0%    | 86.7%    | 1.1s           |
| GPT-3     | 84.5%    | 90.2%    | 1.5s           |

> **Note:** Our custom fine-tuned RoBERTa model demonstrated improved performance over its base variant, showing the impact of domain-specific tuning.

---

## 📁 Folder Structure

```
│── train.csv
│── test.csv
├── QA_cleaned.ipynb
├── README.md
```

---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/qa-roberta.git
   cd qa-roberta
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run fine-tuning (or load the pretrained model):
   ```python
   model = AutoModelForQuestionAnswering.from_pretrained('./model/roberta-finetuned')
   ```

4. Make predictions:
   ```python
   predict_answer(context, question)
   ```

---

## ✅ Future Work

- Optimize inference time using quantization and ONNX.
- Explore deployment using FastAPI or Streamlit.
- Implement multi-hop and open-domain QA.

---

## 📚 References

- [SQuAD Dataset](https://rajpurkar.github.io/SQuAD-explorer/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [RoBERTa Paper](https://arxiv.org/abs/1907.11692)
