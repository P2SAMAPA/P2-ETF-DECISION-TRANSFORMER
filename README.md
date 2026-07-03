# 🎯 P2-ETF-DECISION-TRANSFORMER

**Decision Transformer Engine (Offline RL) — Chen et al. (2021)**

Part of the **P2Quant Engine Suite** · [P2SAMAPA](https://github.com/P2SAMAPA)

---

## What This Engine Does

This engine reframes trading as **offline reinforcement learning via sequence
modelling**. Instead of learning a value function or policy through repeated
environment interaction (as DQN-ENGINE, ERL-ENGINE, and CAUSAL-RL do), it
trains a causal transformer to predict actions from a fixed, already-collected
historical tape of (Return-to-go, State, Action) triples — pure behavior
cloning, no online interaction, no sample-efficiency problem.

At inference, the model is conditioned on a **desired high return** and asked:
*"what action would a trajectory achieving that return have taken here?"* —
turning a sequence predictor into a return-conditioned trading policy.

---

## Theory

### RL as Sequence Modelling

| | Standard Online RL (DQN/ERL/CAUSAL-RL) | **Decision Transformer (this engine)** |
|---|---|---|
| Learning signal | Bellman TD error / fitness / causal effect | **Behavior cloning on fixed tape** |
| Requires environment interaction | Yes | **No — fully offline** |
| Sample efficiency | Bottlenecked by online collection | **Trains on all available history at once** |
| Output | Value function → derived policy | **Direct sequence-to-action prediction** |

### Trajectory Representation

```
trajectory = (R_1, s_1, a_1, R_2, s_2, a_2, ..., R_K, s_K, a_K)
```

- **R_t** — return-to-go: `sum_{k=t}^{T} gamma^{k-t} * r_k`
- **s_t** — state: normalized lagged returns + macro context
- **a_t** — action: position taken, `a_t ∈ [-1, +1]` via tanh

### Offline Dataset Construction

No literal trade log exists, so a hindsight-optimal offline tape is
synthesized directly from realized returns — the standard approach for
building financial offline-RL datasets:

```
a_t = tanh(ACTION_SCALE * forward_return_t)      (behavior-cloning target)
r_t = a_t * forward_return_t                       (reward achieved)
R_t = sum_{k=t}^{T} r_k                             (return-to-go, GAMMA=1)
```

### Architecture

A minimal causal (GPT-style) transformer, **built from scratch with manual
forward/backward passes** (no autograd framework), matching this suite's
established from-scratch modelling pattern:

- separate linear embeddings for R, s, a tokens + learned positional embedding
  per timestep; sequence = `[R_1,s_1,a_1, ..., R_K,s_K,a_K]` (3·CONTEXT_LEN tokens)
- N_LAYERS causal multi-head self-attention blocks + feed-forward, pre-LayerNorm,
  residual connections
- causal mask ensures the representation at each `s_t` position has only seen
  tokens up to and including `s_t` — `a_t` is never leaked
- linear head on the `s_t` representation predicts `a_t`

Trained via Adam with fully analytical gradients through attention,
layernorm, and the feed-forward blocks — no finite differences.

### Conditioning on Desired Return

```
"What action would a trajectory achieving a HIGH return-to-go take here?"
```

Feed in an ambitious `R_t` (90th percentile of the in-sample return-to-go
distribution) and read the model's recommended action directly — the
return-conditioned policy.

### Score Construction

```
score = 0.50 * action_signal  +  0.25 * rtg_sensitivity  +  0.25 * insample_edge
```

| Component | Meaning |
|-----------|---------|
| action_signal | Predicted position when conditioned on a HIGH desired return |
| rtg_sensitivity | action_signal minus the action predicted at a LOW desired return — measures how responsive the policy actually is to what it's asked for |
| insample_edge | Correlation between predicted actions and realized forward returns across the training window — validates the policy tracks genuinely profitable behavior, not noise |

---

## Distinction from Other RL Engines

| Engine | Learning signal | Requires environment interaction |
|--------|------------------|-----------------------------------|
| DQN-ENGINE | Bellman TD error | Yes (online) |
| ERL-ENGINE | Evolutionary fitness | Yes (online) |
| CAUSAL-RL | Causal effect estimation | Yes (online) |
| **DT (this engine)** | **Behavior cloning on fixed tape** | **No (offline)** |

The key distinction: DT never interacts with — or simulates interacting
with — an environment. It sees the historical tape exactly once, as a
supervised sequence-prediction problem, avoiding the online sample-
efficiency problem entirely.

---

## Universes & Windows

| Universe | Tickers |
|---|---|
| FI_COMMODITIES | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| EQUITY_SECTORS | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM, IWD, IWO, XLB, XLRE |
| COMBINED | All of the above |

**Windows:** `63d · 126d · 252d · 504d`

---

## Repository Structure

```
P2-ETF-DECISION-TRANSFORMER/
├── config.py          # Universes, transformer hyperparameters, score weights
├── data_manager.py    # HuggingFace loader
├── dt_engine.py        # Core: causal transformer, analytical backprop, offline RL training
├── trainer.py          # Orchestrator
├── push_results.py     # HfApi.upload_file wrapper
├── streamlit_app.py     # Two-tab Streamlit dashboard
├── us_calendar.py      # US trading calendar helper
├── requirements.txt
└── .github/
    └── workflows/
        └── daily.yml   # Single job
```

---

## Setup

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-DECISION-TRANSFORMER
cd P2-ETF-DECISION-TRANSFORMER
pip install -r requirements.txt

export HF_TOKEN=hf_...
python trainer.py
streamlit run streamlit_app.py
```

**Required GitHub secret:** `HF_TOKEN`

**Required HuggingFace dataset repo:** `P2SAMAPA/p2-etf-decision-transformer-results`

---

## References

- Chen, L. et al. (2021). Decision Transformer: Reinforcement Learning via
  Sequence Modeling. NeurIPS 2021.
- Janner, M., Li, Q. & Levine, S. (2021). Offline Reinforcement Learning as
  One Big Sequence Modeling Problem. NeurIPS 2021.
- Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS 2017.
