```text
███╗   ██╗ ██████╗ ███████╗███╗   ██╗        ████╗
████╗  ██║██╔════╝ ██╔════╝████╗  ██║          ██║
██╔██╗ ██║██║  ███╗█████╗  ██╔██╗ ██║          ██║
██║╚██╗██║██║   ██║██╔══╝  ██║╚██╗██║ ██████╗  ██║
██║ ╚████║╚██████╔╝███████╗██║ ╚████║ ╚═════╝  ██║
╚═╝  ╚═══╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝          ╚═╝ 
```


# ARCH-X Transformer (TensorFlow Implementation)

This project implements a complete **Transformer-based decoder-only architecture** named **ARCH-X**, built from scratch using TensorFlow/Keras. It includes:
- A custom tokenizer
- A multi-head attention mechanism
- Feedforward networks
- Positional encoding
- A full decoder stack
- A text-generation loop
- A benchmarked training pipeline on WikiText-2

---

## 📘 Overview
ARCH-X is modeled after modern decoder-only LLMs. It trains on the WikiText-2 dataset to generate text and demonstrates:
- Manual architecture construction
- End-to-end training pipeline
- Custom tokenization
- Sampling-based autoregressive generation

This is ideal for learning how transformer LMs work under the hood.

---

## 🔧 Key Components
### ✔ Custom Tokenizer (XTokenizer)
A simple tokenizer built using `tf.keras.preprocessing.text.Tokenizer` to:
- Fit vocabulary
- Convert text to sequences
- Pad sequences
- Reverse tokens back to text

### ✔ Decoder-Only Transformer
Includes:
- Multi-Head Self-Attention
- Feedforward network
- Layer normalization
- Dropout
- Positional encoding

### ✔ Training Pipeline
- WikiText-2 dataset
- Next-token prediction target shifting
- Adam optimizer
- SparseCategoricalCrossentropy loss

### ✔ Text Generation
- Categorical sampling
- Temperature scaling
- Autoregressive decoding

---

## 🚀 Running the Model
### 1. Install dependencies:
```bash
pip install tensorflow datasets tqdm numpy
```

### 2. Run training:
Simply execute the script.
Training:
- Trains for 4 epochs
- Uses batch size: 12
- Prints accuracy and loss
- Benchmarks epoch duration

---

## 📊 Model Hyperparameters
- **vocab_size**: dynamic from dataset
- **d_model**: 768
- **num_layers**: 12
- **num_heads**: 12
- **dff**: 3072
- **max_length**: 128
- **dropout_rate**: 0.1

These values correspond to a mid-sized transformer similar to GPT-2 Medium.

---

## 💾 Saving & Loading
The script saves:
- Full model → `arch_x_model/`
- Weights → `arch_x_model_weights.h5`

And reloads it with custom objects.

---

## 🧠 Text Generation Example
Using the reloaded model:
```
start_string = "Nachiketh is good boy and"
```
Generates 100 tokens using temperature **7.0**.

---

## 📁 Use Cases
- Educational transformer implementation
- Lightweight NLP research
- Pretraining experiments
- Understanding decoder-only models
- Benchmarking TensorFlow transformer speed

---

## 🏁 Summary
This script provides:
- A complete working LLM-style architecture
- A real training run on WikiText-2
- Proper save/load functionality
- A text generator

ARCH-X is the perfect base for experimenting with: pretraining, scaling, modifying attention blocks, or testing new transformer research ideas.
