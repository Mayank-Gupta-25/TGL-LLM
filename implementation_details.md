# TGL-LLM Project Overview: Model Implementations & Pipelines

This document provides a comprehensive textual explanation of the three core model architectures implemented in this repository, detailing what was provided by external resources, the specific pipelines and parameters utilized during training/evaluation, and the architectural differences between them.

---

## 1. REGCN (Relational Evolving Graph Convolutional Network)
**Purpose:** A baseline Temporal Knowledge Graph (TKG) forecasting model that predicts future events purely based on historical graph snapshots. It relies purely on graph neural network architecture (without Large Language Models).

### Implementation Pipeline:
- **Architecture:** `REGCN` (which internally builds upon convolutional translations like `convtranse`). It takes historical KG snapshots of length `hist_len` and aggregates them to predict the next query.
- **Workflow (`train.py`):**
  1. Loads temporal graphs per dataset and splits them by time (`train`, `valid`, `test`).
  2. For a sequence of length `hist_len`, it processes the graphs via GNN Message Passing (`uvrgcn` by default).
  3. Uses a decoder (`convtranse`) to calculate raw scores for each potential object entity, ranking them to calculate Hits@K and MRR metrics.
- **Pre-existing vs. Trained:** The entire REGCN structural configuration and PyTorch setup was trained from scratch in this repository on our dataset graphs `[IR, IS, EG]`.
- **Training Parameters (`config_pretrain.yaml` & `config.yaml`):**
  - **Epochs:** `n_epochs: 100` (Defined in logic, although script config defaults some to 0 depending on the run mode).
  - **Learning Rate / Weight Decay:** `lr: 0.001`, `wd: 1e-6`
  - **Network Dimensionality:** `h_dim: 200`
  - **Message Passing Layers:** `n_layers: 2`
  - **Dropout:** Graph MSG dropout, Input, Hidden, and Feature dropout all set to `0.2`
  - **History Length:** `hist_len: 3` (for IR and IS) and `hist_len: 1` (for EG)
  - **Gradient Clipping:** `grad_norm: 1.0`
  - **Patience:** Early stopping triggered after `5` epochs without MRR improvement.

---

## 2. Raw LLM (Language Model - Ablation Baseline)
**Purpose:** A text-only baseline testing the latent reasoning ability of a Large Language Model on Temporal KG problems without any graph structures injected into it.

### Implementation Pipeline:
- **Architecture:** `Llama-2-7b-chat-hf` adapted with LoRA for parameter-efficient fine-tuning on multiple-choice prompting.
- **Workflow (`train_raw_llm.py` & `test_raw_llm.py`):**
  1. Maps Graph IDs back to English text elements: `(Subject, Relation, ?)`.
  2. Constructs a Multiple Choice text prompt: *"Given the historical context, what is the most likely Object Entity for the Query(Sub, Rel, ?) ... A. Obj1  B. Obj2  C. Obj3..."*
  3. Passes text only into the LLM. The LLM is trained to generate the correct letter ("A", "B", etc.) corresponding to the golden target.
- **Pre-existing vs. Trained:**
  - **Given:** The base 7 Billion parameter `meta-llama/Llama-2-7b-chat-hf` weights.
  - **Trained:** A LoRA adapter was trained for each dataset and K-value (e.g., K=3, K=9) to teach the model to adhere to the `A/B/C/D` format based on historical context phrasing.
- **Training Parameters (`train_raw_llm.py`):**
  - **Epochs:** `1`
  - **Batch Size:** `bs_train: 32` (with gradient accumulation to simulate `128`)
  - **Learning Rate:** `3e-4`
  - **LoRA Configuration:** `r=8`, `lora_alpha=16`, `lora_dropout=0.05`, targeting `q_proj` and `v_proj`.
  - **Precision:** `bfloat16`
  - **Loss Function:** Standard Causal Language Modeling objective (`CrossEntropyLoss`).

---

## 3. TGL-LLM (Temporal Graph Learning - Large Language Model)
**Purpose:** The primary novel architecture of this repository. It fuses the structural representation capability of REGCN with the semantic reasoning capability of Llama-2.

### Implementation Pipeline:
- **Architecture:** Deep integration where the hidden outputs of **REGCN** are passed through a **Linear Projector** to map graph embedding dimensionality to the LLM's token embedding dimensionality.
- **Workflow (`train_llm.py`):**
  1. Computes the historical context embeddings using the pre-trained REGCN frozen model.
  2. A linear projector (`projector.bin`) shifts the `200-dim` graph embeddings into the Llama-2 representation space (`4096-dim`).
  3. These transformed graph vectors are interleaved directly into the LLM's input embedding sequence as "soft prompt" tokens.
  4. The LLM receives `[Soft Graph Tokens] + [Text Prompt]` to generate the forecasting object.
- **Pre-existing vs. Trained:**
  - **Given:** Pre-trained `REGCN` from phase 1, and Pre-trained `Llama-2-7b-chat-hf` weights from HuggingFace.
  - **Trained:** The Linear Projector weights and the LoRA adapters for the LLM. 
- **Training Parameters (`config.yaml` & `train_llm.py`):**
  - **Epochs:** `train_epoch: 1`
  - **Batch Size:** `128` (implemented as `batch_size_train=32` with Gradient Accumulation = 4).
  - **Learning Rate:** `3e-4` using `adamw_torch` optimizer.
  - **LoRA Config:** (Inherits raw parameters) `r=8`, `lora_alpha=16`
  - **Soft Prompts:** Enabled via `projector_evo.bin` and `prompt_token.bin` mappings in `TKGLLMEVO`.
  - **Generation Config:** Greedy decoding (`do_sample=False`, `max_new_tokens=5`, `num_beams=1`)

---

### Conclusion & Pipeline Summary
The full pipeline flows consecutively:
1. **Pre-Training Phase (`train.py`)**: The REGCN extracts the topological structures of the graph and learns structural embeddings iteratively up to `100` epochs.
2. **Alignment Phase (`train_llm.py`)**: The TGL-LLM freezes both the REGCN and LLM and trains *only* the middle linear projector network to map graph-space to language-space.
3. **Fine-Tuning Phase (`train_llm.py`)**: LoRA is applied to the Llama model alongside the projector, training both together for 1 epoch to harmonize graph reasoning with text generation.
4. **Ablation (`train_raw_llm.py`)**: Independent of the pipeline, LoRA trains the Raw LLM on pure text to prove that without Phase 1 and 2, accuracy drops significantly.
