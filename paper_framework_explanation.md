# TGL-LLM: Framework Explanation & Paper Analysis

## 1. Problem Statement

**Temporal Knowledge Graph Forecasting (TKGF)** aims to predict future events based on observed historical events. Events are represented as quadruples: `(Subject, Relation, Object, Timestamp)` — e.g., `(Israel, Threaten, Iran, 2024-01-15)`.

**Why is this hard?**
- Traditional graph models (RGCN, REGCN) are limited to training data and struggle with rare/long-tail entities.
- LLMs have broad world knowledge but cannot natively understand graph structures.
- Existing LLM-based methods either use **text-based** retrieval (which can miss critical context and hit token limits) or **static graph embeddings** (which ignore temporal dynamics).

**TGL-LLM's Goal:** Bridge temporal graph learning and LLMs by injecting *time-aware* graph embeddings directly into the LLM's token space.

---

## 2. Framework Architecture (Figure 2)

The TGL-LLM framework has **three main components**:

### 2.1 Temporal Graph Learning (Left side of Figure 2)

This is the **REGCN** pre-training stage. It processes raw temporal knowledge graphs to produce numerical embeddings for every entity and relation.

**Step-by-step:**
1. **Input:** Historical graph snapshots `{G(t-2), G(t-1), G(t)}`, where each snapshot contains all events at that timestamp.
2. **RGCN Layers (×2):** At each timestamp, a 2-layer Relational Graph Convolutional Network aggregates messages from neighboring entities. Each entity gets a 200-dimensional embedding vector.
   - Formula: `e_o = f(Σ W1·(e_s + r) + W2·e_o)` for each `(s, r, o)` in the graph.
3. **GRU (Training only):** A Gated Recurrent Unit captures temporal evolution patterns across timestamps: `Ê_t = GRU(E_t, E_{t-1})`.
4. **ConvTransE Decoder (Training only):** Scores all possible object entities for a given `(subject, relation, ?)` query using convolutional operations.
5. **Output:** Entity embeddings `E_t ∈ R^{|E|×200}` and relation embeddings `R ∈ R^{|R|×200}` for the most recent T=3 timesteps.

**Key Insight:** During *inference*, only the RGCN layer outputs (the per-timestamp entity embeddings) are passed to the LLM — **not** the GRU or ConvTransE outputs. This preserves raw temporal information for the LLM to reason over, rather than giving it a pre-digested summary.

### 2.2 Hybrid Graph Tokenization (Middle of Figure 2)

This is the bridge between the graph world (200-dim vectors) and the LLM world (4096-dim token embeddings).

**Components:**
1. **Entity Adapter (EA):** A 2-layer MLP that projects entity embeddings from `R^200 → R^4096` (Llama-2's hidden dimension).
2. **Relation Adapter (RA):** A separate 2-layer MLP that projects relation embeddings from `R^200 → R^4096`.
3. **Feature Token `<f>`:** A special learnable soft prompt token inserted between text tokens and graph tokens, helping the LLM distinguish between the two modalities.

**Hybrid Prompt Construction:**  
The final prompt fed to the LLM is a sequence of mixed text embeddings and graph embeddings:

```
[BOS] [Text: "Given the historical context..."] [<f>] [Graph: subject_t-2, subject_t-1, subject_t]
[<f>] [Graph: relation]
[Text: "A."] [<f>] [Graph: candidate_A_t-2, candidate_A_t-1, candidate_A_t]
[Text: "B."] [<f>] [Graph: candidate_B_t-2, candidate_B_t-1, candidate_B_t]
...
[Text: "### Response:"]
```

This is called **"hybrid"** because each entity is described by both its text name AND its temporal graph embeddings from the last 3 timesteps.

### 2.3 Model Training — Two-Stage Paradigm (Right side of Figure 2)

Instead of training on all data at once, TGL-LLM uses a clever two-stage approach:

**Data Pruning (before training):**
1. **Influence Function Scoring:** Uses the pre-trained REGCN as a surrogate model to calculate how much each training sample matters. Samples with higher influence scores are "higher quality" for cross-modal alignment.
2. **Stratified Sampling:** Selects a **high-quality subset** `D_h` (100,000 samples) from the scored data.
3. **Diversity Sampling:** Randomly selects a smaller **diversity subset** `D_p` (10,000 samples) to ensure the LLM sees varied temporal and relational patterns.

**Stage 1 — High-Quality Alignment:**
- Fine-tune the LLM (with LoRA) on the high-quality subset `D_h`.
- This teaches the model to align graph embeddings with language tokens.
- Trainable: LoRA weights, Entity Adapter, Relation Adapter, Feature Token.
- Frozen: Base LLM weights, REGCN weights.

**Stage 2 — Diversity Fine-Tuning:**
- Continue fine-tuning on the diversity subset `D_p`.
- This improves the model's generalization across different pattern types.

---

## 3. Implementation Details (from Codebase)

### 3.1 File-to-Component Mapping

| Component | File | Purpose |
|-----------|------|---------|
| Graph Generation | `generate_graphs.py` | Converts raw `.txt` quadruples into DGL graph objects |
| REGCN Model | `modules/regcn.py` | RGCN + GRU + ConvTransE decoder |
| REGCN Training | `train.py` | Pre-trains the temporal graph model |
| Data Pruning | `prune.py` | Influence function scoring + coreset selection |
| TGL-LLM Model | `modules/tglllm.py` | Hybrid prompt construction + Llama-2 integration |
| LLM Training/Testing | `train_llm.py` | LoRA fine-tuning and MCQ evaluation |
| Data Utilities | `modules/utils_llm.py` | Dataset loading, candidate management |
| ConvTransE Decoder | `modules/decoder.py` | Scoring function for REGCN predictions |

### 3.2 Pipeline Execution Order

```
1. generate_graphs.py  →  Creates graph_dict.pkl
2. train.py            →  Pre-trains REGCN (saves checkpoint)
3. prune.py            →  Scores training data, creates coreset JSON
4. train_llm.py -o train           →  Stage 1 + Stage 2 LLM fine-tuning
5. train_llm.py -o test -k <K>     →  MCQ evaluation at K choices
```

### 3.3 Key Hyperparameters

| Parameter | Value | Where |
|-----------|-------|-------|
| Graph embedding dim (`h_dim`) | 200 | `config_pretrain.yaml` |
| RGCN layers (`n_layers`) | 2 | `config_pretrain.yaml` |
| History length (`hist_len`) | 3 timesteps | `config.yaml` |
| REGCN learning rate | 0.001 | `config_pretrain.yaml` |
| REGCN epochs | 30 | `config_pretrain.yaml` |
| LLM backbone | Llama-2-7b-chat-hf | `config.yaml` |
| LLM learning rate | 3e-4 | `train_llm.py` |
| LLM training epochs | 1 | `config.yaml` |
| LoRA rank (`r`) | 8 | `modules/tglllm.py` |
| LoRA alpha | 16 | `modules/tglllm.py` |
| LoRA target modules | q_proj, v_proj | `modules/tglllm.py` |
| Batch size (effective) | 128 (32 × 4 grad accum) | `train_llm.py` |
| Coreset size (Stage 1) | 100,000 | `prune.py` |
| Diversity subset (Stage 2) | 10,000 | `prune.py` |

---

## 4. Figures Explained

### Figure 1: Limitations of Existing Methods

**What it shows:** A side-by-side comparison of the two existing LLM-based TKGF approaches and their failure modes.

- **(a) Text-based:** Shows that retrieval-augmented methods can miss critical historical facts (e.g., `(Iran, aid, Palestine, t-1)` is omitted) and face token length limits. The LLM responds: *"Sorry, the history provided is insufficient."*
- **(b) Embedding-based:** Shows that static graph embeddings (which collapse all temporal information into one snapshot) cause the LLM to give the wrong answer (`China` instead of `Iran`), because it cannot distinguish between events at different timestamps.

**Takeaway:** Both approaches fail — text loses structural info, static embeddings lose temporal info. TGL-LLM solves this by using *temporal* graph embeddings.

### Figure 2: TGL-LLM Overall Framework

**What it shows:** The complete architecture diagram with three panels:
- **Left:** Temporal Graph Learning — RGCN processes graph snapshots at each timestamp, producing entity/relation embeddings.
- **Middle:** Hybrid Graph Tokenization — Entity/Relation Adapters project graph embeddings into LLM token space. The hybrid prompt interleaves text tokens with graph tokens.
- **Right:** Model Training — Two-stage paradigm with data pruning via influence functions.

**Key visual elements:**
- Blue blocks = frozen/pre-trained components
- Orange blocks = trainable components (LoRA, Adapters, Feature tokens)
- `<f>` markers = learned feature tokens separating text from graph modalities

### Figure 3: Long-Tail Entity Performance

**What it shows:** Bar charts comparing Acc@4 and Acc@10 across "Dense" (frequent) and "Sparse" (long-tail) entity subsets on POLECAT-IR and POLECAT-EG.

**Key findings:**
- LLM-based methods (CoH, TGL-LLM) outperform Non-LLM methods (HisMatch) on **sparse/long-tail** entities because the LLM brings external world knowledge.
- On **dense** entities, Non-LLM methods remain competitive since they have enough training data.
- TGL-LLM achieves the best performance on **both** dense and sparse subsets with the smallest gap between them.

### Figure 4: Impact of History Length

**What it shows:** Line plots of Acc@10 as the historical graph window varies from 1 day to 7 days across all three datasets.

**Key findings:**
- Optimal history length is **5 days** for POLECAT-IR and POLECAT-IS.
- Optimal history length is **7 days** for POLECAT-EG.
- Performance converges and slightly degrades with very long histories, suggesting that too much history introduces noise.

---

## 5. Tables Explained

### Table 1: Dataset Statistics

| Set | POLECAT-IR | POLECAT-IS | POLECAT-EG |
|-----|------------|------------|------------|
| **Train** | 616,880 facts, 33,920 entities | 484,630 facts, 31,898 entities | 256,523 facts, 25,422 entities |
| **Valid** | 13,737 facts | 29,931 facts | 6,898 facts |
| **Test** | 11,053 facts | 56,750 facts | 3,812 facts |
| **Relations** | 80 | 80 | 80 |

**Split strategy:** Training = Jan 2018–May 2023, Validation = Jun–Oct 2023, Test = Nov 2023–Apr 2024. The test set is **after** Llama-2's training cutoff (July 2023), preventing data leakage.

### Table 2: Main Performance Comparison (Acc@4, Acc@6, Acc@10)

This is the **primary results table** comparing all methods across three datasets.

**Non-LLM Methods (Graph input only):**
- DistMult, ConvTransE, RGCN, RENET, REGCN, HisMatch
- Best non-LLM: **HisMatch** on IR and IS, **ConvTransE** on EG

**Zero-Shot LLM Methods (Text input only):**
- GPT-3.5-turbo, GPT-4o-mini
- Surprisingly weak — near random chance on some metrics

**Fine-Tuned LLM Methods:**
- GenTKG, CoH (text-based fine-tuning)
- KoPA (static graph embedding fine-tuning)
- **TGL-LLM (proposed):** Best across ALL datasets and ALL K values

**Key result:** TGL-LLM achieves **85.14% Acc@4** on IR vs the next best method KoPA at **60.83%**, a **24+ percentage point improvement**.

### Table 3: Ablation — Impact of Graph Representation Type

Compares what happens when you vary *which layer* of the graph model feeds into the LLM:

| Variant | What it feeds to LLM | IR Acc@4 | IR Acc@10 |
|---------|---------------------|----------|-----------|
| **Raw** | No graph data (text only) | 56.88% | 35.10% |
| **Static** | Static graph embedding (no temporal info) | 61.65% | 40.84% |
| **GRU** | GRU output (deep temporal) | 70.98% | 64.04% |
| **ConvTransE** | ConvTransE output (deepest) | 80.80% | 62.30% |
| **TGL-LLM** | Recent RGCN embeddings (shallow temporal) | **85.14%** | **74.07%** |

**Key insight:** Deeper graph representations (GRU, ConvTransE) actually perform *worse* than TGL-LLM's shallower RGCN embeddings. The authors hypothesize that heavily processed embeddings are harder for the LLM to align with its token space.

### Table 4: Ablation — Impact of Sampling Strategy

Tests the two-stage training paradigm:

| Variant | Description | IR Acc@4 |
|---------|------------|----------|
| **Random** | Random sampling, no pruning | 84.67% |
| **w/o-IF** | Without influence function (random quality) | 83.69% |
| **w/o-DS** | Without diversity subset (Stage 2 skipped) | 83.54% |
| **TGL-LLM** | Full two-stage (quality + diversity) | **85.14%** |

**Key insight:** Both the high-quality subset AND the diversity subset contribute to performance. Removing either one degrades results.

### Table 5: Training Cost Comparison

| Method | Avg. Token Length | GPU Memory (GiB) | Training Time (hours) | Acc@10 |
|--------|-------------------|-------------------|----------------------|--------|
| **CoH** | 466 | 36.78 | 21.53 | 0.406 |
| **KoPA** | 223 | 22.98 | 10.23 | 0.4358 |
| **TGL-LLM** | 245 | 23.08 | 11.26 | **0.7407** |

**Key insight:** TGL-LLM uses slightly more resources than KoPA but achieves **70%+ better Acc@10**. It is far more efficient than text-based CoH while being dramatically more accurate.

---

## 6. Key Innovations Summary

| Innovation | Problem Solved | How |
|-----------|---------------|-----|
| **Hybrid Graph Tokenization** | Insufficient temporal modeling in LLMs | Injects per-timestep RGCN embeddings as pseudo-tokens via learned adapters |
| **Two-Stage Training** | Ineffective graph-language alignment | Stage 1: high-quality data alignment. Stage 2: diversity for generalization |
| **Influence Function Pruning** | Noisy/redundant training data | Scores each sample's contribution using Hessian-based influence analysis |
| **Shallow Graph Tokens** | Deep embeddings hard to align | Uses RGCN output (not GRU/ConvTransE) for better LLM compatibility |
