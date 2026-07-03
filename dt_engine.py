"""
dt_engine.py — Decision Transformer Engine (Offline RL)
==========================================================

Theory
------
**Reinforcement Learning as Sequence Modelling (Chen et al. 2021)**

Standard RL (DQN-ENGINE, ERL-ENGINE, CAUSAL-RL in this suite — all ONLINE)
learns a value function or policy through repeated interaction with an
environment: act, observe reward, update, repeat. This requires either a
live environment or a simulator, and suffers from the sample-efficiency
problem — many interactions are needed before the policy is any good.

The Decision Transformer instead treats a trajectory of

    (R_1, s_1, a_1, R_2, s_2, a_2, ..., R_K, s_K, a_K)

as a single sequence, where:
    R_t = return-to-go at t  = sum_{k=t}^{T} gamma^{k-t} * r_k
    s_t = market state at t  (lagged returns + macro context)
    a_t = action (position) taken at t

and trains a CAUSAL transformer via ordinary behavior cloning: predict a_t
from the causal context (R_1..s_1..a_1, ..., R_t, s_t). No environment
interaction is required at all — the model is trained purely by supervised
sequence prediction on a FIXED, already-collected (offline) tape of
trajectories. This is what makes it "offline": the tape is built once from
historical data and never touched again during training.

**Conditioning on desired return**

The key trick that makes this work as a *policy* rather than just a
predictor: at inference, feed in a manually chosen, ambitious R_t and let
the model tell you what action a trajectory achieving that return would
have taken. The model has learned, purely from the shape of historical
(R,s,a) sequences, what "a policy that achieves high returns" looks like at
each state s_t — it never had to discover this via trial and error.

**Offline dataset construction**

This suite doesn't have literal historical trade logs, so a hindsight-
optimal offline dataset is synthesized directly from realized returns —
the standard approach for building financial offline-RL datasets:

    a_t = tanh(ACTION_SCALE * forward_return_t)      (behavior-cloning target)
    r_t = a_t * forward_return_t                       (reward achieved)
    R_t = sum_{k=t}^{T} r_k                             (return-to-go, GAMMA=1)

**Architecture**

A minimal causal (GPT-style) transformer, built from scratch with manual
forward/backward passes (no autograd framework), matching this suite's
established from-scratch modelling pattern:
    - separate linear embeddings for R, s, a tokens + learned positional
      embedding per timestep, sequence = [R_1,s_1,a_1, ..., R_K,s_K,a_K]
      (3*CONTEXT_LEN tokens)
    - N_LAYERS transformer blocks: causal multi-head self-attention +
      residual, feed-forward (tanh) + residual, pre-LayerNorm
    - causal mask ensures the representation at each s_t position has only
      seen tokens up to and including s_t — a_t is never leaked
    - linear head on the s_t representation predicts a_t

**Score construction**

    score = 0.50 * action_signal + 0.25 * rtg_sensitivity + 0.25 * insample_edge

| Component        | Meaning                                                        |
|-------------------|-----------------------------------------------------------------|
| action_signal     | Predicted position when conditioned on a HIGH desired return    |
| rtg_sensitivity   | action_signal minus action predicted at a LOW desired return    |
| insample_edge     | Correlation between predicted actions and realized fwd returns  |

**Key distinction from other RL engines in the suite**

| Engine       | Learning signal source          | Requires environment interaction |
|--------------|----------------------------------|-----------------------------------|
| DQN-ENGINE   | Bellman TD error                 | Yes (online)                      |
| ERL-ENGINE   | Evolutionary fitness              | Yes (online)                      |
| CAUSAL-RL    | Causal effect estimation          | Yes (online)                      |
| **DT (this)**| **Behavior cloning on fixed tape**| **No (offline)**                  |

References
----------
- Chen, L. et al. (2021). Decision Transformer: Reinforcement Learning via
  Sequence Modeling. NeurIPS 2021.
- Janner, M., Li, Q. & Levine, S. (2021). Offline Reinforcement Learning as
  One Big Sequence Modeling Problem. NeurIPS 2021.
- Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS 2017.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

import config


# ── Basic differentiable layers (manual forward/backward) ─────────────────────

class Linear:
    def __init__(self, in_d: int, out_d: int, rng: np.random.Generator):
        scale = np.sqrt(2.0 / in_d)
        self.W = rng.normal(0, scale, (in_d, out_d))
        self.b = np.zeros(out_d)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X
        return X @ self.W + self.b

    def backward(self, dY: np.ndarray):
        X = self.X
        X2  = X.reshape(-1, X.shape[-1])
        dY2 = dY.reshape(-1, dY.shape[-1])
        dW  = X2.T @ dY2
        db  = dY2.sum(axis=0)
        dX  = dY @ self.W.T
        return dX, dW, db


class LayerNorm:
    def __init__(self, d: int):
        self.gamma = np.ones(d)
        self.beta  = np.zeros(d)

    def forward(self, X: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mu  = X.mean(axis=-1, keepdims=True)
        var = X.var(axis=-1, keepdims=True)
        std = np.sqrt(var + eps)
        Xhat = (X - mu) / std
        self.cache = (Xhat, std)
        return self.gamma * Xhat + self.beta

    def backward(self, dY: np.ndarray):
        Xhat, std = self.cache
        d = Xhat.shape[-1]
        dgamma = (dY * Xhat).reshape(-1, d).sum(axis=0)
        dbeta  = dY.reshape(-1, d).sum(axis=0)
        dXhat  = dY * self.gamma
        dX = (1.0 / std) * (
            dXhat
            - dXhat.mean(axis=-1, keepdims=True)
            - Xhat * (dXhat * Xhat).mean(axis=-1, keepdims=True)
        )
        return dX, dgamma, dbeta


class CausalSelfAttention:
    def __init__(self, D: int, H: int, rng: np.random.Generator):
        self.D, self.H, self.Dh = D, H, D // H
        self.Wq = Linear(D, D, rng)
        self.Wk = Linear(D, D, rng)
        self.Wv = Linear(D, D, rng)
        self.Wo = Linear(D, D, rng)

    def forward(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        B, T, D = X.shape
        H, Dh = self.H, self.Dh
        Q = self.Wq.forward(X)
        K = self.Wk.forward(X)
        V = self.Wv.forward(X)

        Qh = Q.reshape(B, T, H, Dh).transpose(0, 2, 1, 3)
        Kh = K.reshape(B, T, H, Dh).transpose(0, 2, 1, 3)
        Vh = V.reshape(B, T, H, Dh).transpose(0, 2, 1, 3)

        scores = (Qh @ Kh.transpose(0, 1, 3, 2)) / np.sqrt(Dh)
        scores = scores + mask
        scores = scores - scores.max(axis=-1, keepdims=True)
        expS = np.exp(scores)
        A = expS / expS.sum(axis=-1, keepdims=True)

        Oh = A @ Vh
        O  = Oh.transpose(0, 2, 1, 3).reshape(B, T, D)
        out = self.Wo.forward(O)

        self.cache = (Qh, Kh, Vh, A, B, T)
        return out

    def backward(self, dOut: np.ndarray):
        dO_flat, dWo_W, dWo_b = self.Wo.backward(dOut)
        Qh, Kh, Vh, A, B, T = self.cache
        H, Dh = self.H, self.Dh

        dOh = dO_flat.reshape(B, T, H, Dh).transpose(0, 2, 1, 3)
        dA  = dOh @ Vh.transpose(0, 1, 3, 2)
        dVh = A.transpose(0, 1, 3, 2) @ dOh

        dScores = A * (dA - (dA * A).sum(axis=-1, keepdims=True))
        dScores = dScores / np.sqrt(Dh)

        dQh = dScores @ Kh
        dKh = dScores.transpose(0, 1, 3, 2) @ Qh

        dQ = dQh.transpose(0, 2, 1, 3).reshape(B, T, H * Dh)
        dK = dKh.transpose(0, 2, 1, 3).reshape(B, T, H * Dh)
        dV = dVh.transpose(0, 2, 1, 3).reshape(B, T, H * Dh)

        dXq, dWq_W, dWq_b = self.Wq.backward(dQ)
        dXk, dWk_W, dWk_b = self.Wk.backward(dK)
        dXv, dWv_W, dWv_b = self.Wv.backward(dV)
        dX = dXq + dXk + dXv

        grads = {
            "Wq": (dWq_W, dWq_b), "Wk": (dWk_W, dWk_b),
            "Wv": (dWv_W, dWv_b), "Wo": (dWo_W, dWo_b),
        }
        return dX, grads


class FeedForward:
    def __init__(self, D: int, hidden: int, rng: np.random.Generator):
        self.L1 = Linear(D, hidden, rng)
        self.L2 = Linear(hidden, D, rng)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.Z1 = self.L1.forward(X)
        self.H1 = np.tanh(self.Z1)
        return self.L2.forward(self.H1)

    def backward(self, dOut: np.ndarray):
        dH1, dW2, db2 = self.L2.backward(dOut)
        dZ1 = dH1 * (1 - self.H1 ** 2)
        dX, dW1, db1 = self.L1.backward(dZ1)
        return dX, {"L1": (dW1, db1), "L2": (dW2, db2)}


class Block:
    def __init__(self, D: int, H: int, FF: int, rng: np.random.Generator):
        self.ln1  = LayerNorm(D)
        self.attn = CausalSelfAttention(D, H, rng)
        self.ln2  = LayerNorm(D)
        self.ffn  = FeedForward(D, FF, rng)

    def forward(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        A  = self.attn.forward(self.ln1.forward(X), mask)
        X1 = X + A
        F  = self.ffn.forward(self.ln2.forward(X1))
        X2 = X1 + F
        self.cache = (X, X1)
        return X2

    def backward(self, dX2: np.ndarray):
        dX1 = dX2.copy()
        dLn2out, ffn_grads = self.ffn.backward(dX2)
        dX1_from_ffn, dln2_gamma, dln2_beta = self.ln2.backward(dLn2out)
        dX1 = dX1 + dX1_from_ffn

        dX = dX1.copy()
        dLn1out, attn_grads = self.attn.backward(dX1)
        dX_from_attn, dln1_gamma, dln1_beta = self.ln1.backward(dLn1out)
        dX = dX + dX_from_attn

        grads = {
            "ln1": (dln1_gamma, dln1_beta),
            "attn": attn_grads,
            "ln2": (dln2_gamma, dln2_beta),
            "ffn": ffn_grads,
        }
        return dX, grads


# ── Decision Transformer model ─────────────────────────────────────────────────

class DecisionTransformer:
    """
    Sequence: [R_1,s_1,a_1, R_2,s_2,a_2, ..., R_K,s_K,a_K]  (3K tokens)
    Predicts a_t from the causally-masked representation at the s_t position.
    """

    def __init__(self, state_dim: int, rng: np.random.Generator):
        D, K = config.EMBED_DIM, config.CONTEXT_LEN
        self.D, self.K = D, K
        self.state_dim = state_dim

        self.R_embed = Linear(1, D, rng)
        self.S_embed = Linear(state_dim, D, rng)
        self.A_embed = Linear(1, D, rng)
        self.pos_embed = rng.normal(0, 0.02, (K, D))

        self.blocks = [Block(D, config.N_HEADS, config.FF_HIDDEN, rng)
                        for _ in range(config.N_LAYERS)]
        self.ln_f  = LayerNorm(D)
        self.head  = Linear(D, 1, rng)

        T = 3 * K
        mask = np.triu(np.ones((T, T)), k=1) * -1e9
        self.mask = mask[None, None, :, :]

    def _build_sequence(self, R: np.ndarray, S: np.ndarray, A: np.ndarray) -> np.ndarray:
        """R:(B,K) S:(B,K,state_dim) A:(B,K) -> X:(B,3K,D)"""
        B, K, D = R.shape[0], self.K, self.D
        tok_R = self.R_embed.forward(R[:, :, None]) + self.pos_embed[None, :, :]
        tok_S = self.S_embed.forward(S)             + self.pos_embed[None, :, :]
        tok_A = self.A_embed.forward(A[:, :, None]) + self.pos_embed[None, :, :]

        X = np.zeros((B, 3 * K, D))
        X[:, 0::3, :] = tok_R
        X[:, 1::3, :] = tok_S
        X[:, 2::3, :] = tok_A
        return X

    def forward(self, R: np.ndarray, S: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Returns predicted actions (B, K) — one prediction per s_t position."""
        X = self._build_sequence(R, S, A)
        for blk in self.blocks:
            X = blk.forward(X, self.mask)
        X = self.ln_f.forward(X)
        self._X_final = X

        s_hidden = X[:, 1::3, :]                     # (B, K, D) — s_t positions
        pred = np.tanh(self.head.forward(s_hidden)).squeeze(-1)   # (B, K)
        self._s_hidden = s_hidden
        self._pred = pred
        return pred

    def backward(self, dPred: np.ndarray):
        """dPred: (B,K) gradient w.r.t. predicted actions. Returns param grads."""
        B, K, D = dPred.shape[0], self.K, self.D
        dTanh = dPred * (1 - self._pred ** 2)
        dHead_in, dHead_W, dHead_b = self.head.backward(dTanh[:, :, None])

        dX = np.zeros((B, 3 * K, D))
        dX[:, 1::3, :] = dHead_in

        dX, dlnf_gamma, dlnf_beta = self.ln_f.backward(dX)

        block_grads = []
        for blk in reversed(self.blocks):
            dX, g = blk.backward(dX)
            block_grads.append(g)
        block_grads.reverse()

        dtok_R = dX[:, 0::3, :]
        dtok_S = dX[:, 1::3, :]
        dtok_A = dX[:, 2::3, :]

        _, dRW, dRb = self.R_embed.backward(dtok_R)
        _, dSW, dSb = self.S_embed.backward(dtok_S)
        _, dAW, dAb = self.A_embed.backward(dtok_A)
        dpos = (dtok_R + dtok_S + dtok_A).sum(axis=0)

        return {
            "R_embed": (dRW, dRb), "S_embed": (dSW, dSb), "A_embed": (dAW, dAb),
            "pos_embed": dpos,
            "blocks": block_grads,
            "ln_f": (dlnf_gamma, dlnf_beta),
            "head": (dHead_W, dHead_b),
        }

    # ── Adam optimizer over the full parameter set ────────────────────────────

    def _param_list(self):
        params = [
            (self.R_embed, "W"), (self.R_embed, "b"),
            (self.S_embed, "W"), (self.S_embed, "b"),
            (self.A_embed, "W"), (self.A_embed, "b"),
            ("pos_embed", None),
            (self.ln_f, "gamma"), (self.ln_f, "beta"),
            (self.head, "W"), (self.head, "b"),
        ]
        for blk in self.blocks:
            params += [
                (blk.ln1, "gamma"), (blk.ln1, "beta"),
                (blk.attn.Wq, "W"), (blk.attn.Wq, "b"),
                (blk.attn.Wk, "W"), (blk.attn.Wk, "b"),
                (blk.attn.Wv, "W"), (blk.attn.Wv, "b"),
                (blk.attn.Wo, "W"), (blk.attn.Wo, "b"),
                (blk.ln2, "gamma"), (blk.ln2, "beta"),
                (blk.ffn.L1, "W"), (blk.ffn.L1, "b"),
                (blk.ffn.L2, "W"), (blk.ffn.L2, "b"),
            ]
        return params

    def init_adam(self):
        state = []
        for obj, attr in self._param_list():
            p = self.pos_embed if obj == "pos_embed" else getattr(obj, attr)
            state.append((np.zeros_like(p), np.zeros_like(p)))
        return state

    def apply_adam(self, grads: dict, state, step: int,
                    lr: float, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        flat_grads = [
            grads["R_embed"][0], grads["R_embed"][1],
            grads["S_embed"][0], grads["S_embed"][1],
            grads["A_embed"][0], grads["A_embed"][1],
            grads["pos_embed"],
            grads["ln_f"][0], grads["ln_f"][1],
            grads["head"][0], grads["head"][1],
        ]
        for bg in grads["blocks"]:
            flat_grads += [
                bg["ln1"][0], bg["ln1"][1],
                bg["attn"]["Wq"][0], bg["attn"]["Wq"][1],
                bg["attn"]["Wk"][0], bg["attn"]["Wk"][1],
                bg["attn"]["Wv"][0], bg["attn"]["Wv"][1],
                bg["attn"]["Wo"][0], bg["attn"]["Wo"][1],
                bg["ln2"][0], bg["ln2"][1],
                bg["ffn"]["L1"][0], bg["ffn"]["L1"][1],
                bg["ffn"]["L2"][0], bg["ffn"]["L2"][1],
            ]

        params = self._param_list()
        for i, ((obj, attr), grad) in enumerate(zip(params, flat_grads)):
            m, v = state[i]
            m[:] = b1 * m + (1 - b1) * grad
            v[:] = b2 * v + (1 - b2) * grad ** 2
            mh = m / (1 - b1 ** step)
            vh = v / (1 - b2 ** step)
            update = lr * mh / (np.sqrt(vh) + eps)
            if obj == "pos_embed":
                self.pos_embed -= update
            else:
                getattr(obj, attr)[:] = getattr(obj, attr) - update


# ── Offline trajectory construction ─────────────────────────────────────────────

def _build_state_features(log_ret: np.ndarray, macro_norm: np.ndarray, L: int) -> np.ndarray:
    """Build state vectors s_t (normalized lagged returns + macro) for all valid t."""
    T = len(log_ret)
    states = []
    for t in range(L, T):
        lag = log_ret[t - L:t]
        mu, std = lag.mean(), lag.std() + 1e-8
        s = np.concatenate([(lag - mu) / std, macro_norm[t]])
        states.append(s)
    return np.array(states)


def _train_decision_transformer(R: np.ndarray, S: np.ndarray, A: np.ndarray,
                                  rng: np.random.Generator) -> DecisionTransformer:
    """R,S,A are full per-timestep arrays; slice into overlapping length-K windows."""
    K = config.CONTEXT_LEN
    N = len(R) - K + 1
    if N < config.DT_BATCH_SIZE:
        raise ValueError("insufficient trajectory length for training")

    model = DecisionTransformer(state_dim=S.shape[1], rng=rng)
    adam_state = model.init_adam()
    B = config.DT_BATCH_SIZE
    step = 0

    for epoch in range(config.DT_EPOCHS):
        idx = rng.permutation(N)
        epoch_loss, n_b = 0.0, 0

        for i in range(0, N, B):
            bi = idx[i:i + B]
            if len(bi) < 2:
                continue

            R_b = np.stack([R[j:j + K] for j in bi])
            S_b = np.stack([S[j:j + K] for j in bi])
            A_b = np.stack([A[j:j + K] for j in bi])
            target = A_b.copy()   # behavior-cloning target = the hindsight action itself

            pred = model.forward(R_b, S_b, A_b)
            resid = pred - target
            loss = float(np.mean(resid ** 2))

            grads = model.backward(2.0 * resid / resid.size)
            step += 1
            model.apply_adam(grads, adam_state, step, lr=config.DT_LR)
            epoch_loss += loss
            n_b += 1

        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1}/{config.DT_EPOCHS}  loss={epoch_loss/max(n_b,1):.6f}")

    return model


# ── Main scoring function ─────────────────────────────────────────────────────

def compute_dt_scores(
    prices:    pd.DataFrame,
    macro_df:  pd.DataFrame,
    tickers:   List[str],
    window:    int,
) -> pd.Series:
    """
    Train a Decision Transformer per ETF on hindsight-optimal offline
    trajectories and extract return-conditioned action scores.

    Returns cross-sectional z-scores.
    """
    avail = [t for t in tickers if t in prices.columns]
    if not avail:
        return pd.Series(dtype=float)

    L, H, K = config.N_LAGS, config.PRED_HORIZON, config.CONTEXT_LEN
    min_rows = window + H + L + K + 5
    if len(prices) < min_rows:
        return pd.Series(dtype=float)

    common   = prices.index.intersection(macro_df.index) if not macro_df.empty else prices.index
    prices_a = prices.loc[common]
    macro_a  = macro_df.loc[common] if not macro_df.empty else pd.DataFrame(index=common)

    macro_vals = macro_a.values.astype(np.float64) if not macro_a.empty else np.zeros((len(common), 0))
    if macro_vals.shape[1] > 0:
        m_mu  = np.nanmean(macro_vals, axis=0, keepdims=True)
        m_std = np.nanstd(macro_vals,  axis=0, keepdims=True) + 1e-8
        macro_norm = np.nan_to_num((macro_vals - m_mu) / m_std, 0.0)
    else:
        macro_norm = np.zeros((len(common), 0))

    rng = np.random.default_rng(42)
    raw_scores = {}

    for ticker in avail:
        ps = prices_a[ticker].dropna()
        if len(ps) < min_rows:
            continue

        log_ret = np.log(ps / ps.shift(1)).dropna().values
        mac = macro_norm[-len(log_ret):]
        if len(mac) < len(log_ret):
            log_ret = log_ret[-len(mac):]

        T = len(log_ret)
        start = max(L, T - window - H)
        end = T - H
        if end - start < K + config.DT_BATCH_SIZE:
            continue

        # ── Build offline (state, action, reward, return-to-go) tape ──────────
        fwd = np.array([log_ret[t:t + H].mean() for t in range(start, end)])
        states = _build_state_features(log_ret[start - L if start >= L else 0:end], mac, L) \
            if start >= L else _build_state_features(log_ret[:end], mac, L)
        # align states to the same t-range as fwd (states computed from index L onward)
        offset = start - L if start >= L else 0
        states = states[max(0, start - offset - L):]
        n = min(len(fwd), len(states))
        fwd, states = fwd[-n:], states[-n:]
        if n < K + config.DT_BATCH_SIZE:
            continue

        actions = np.tanh(config.ACTION_SCALE * fwd)
        rewards = actions * fwd
        rtg = np.cumsum(rewards[::-1])[::-1] * config.GAMMA  # simple undiscounted return-to-go

        print(f"    Training DT for {ticker} (N={n}, state_dim={states.shape[1]})")
        try:
            model = _train_decision_transformer(rtg, states, actions, rng)
        except Exception as e:
            print(f"    Failed {ticker}: {e}")
            continue

        # ── Evaluate: condition on desired HIGH / LOW return-to-go ────────────
        ctx_R = rtg[-K:].copy()
        ctx_S = states[-K:]
        ctx_A = actions[-K:].copy()

        rtg_high = float(np.percentile(rtg, config.TARGET_RTG_PERCENTILE_HIGH))
        rtg_low  = float(np.percentile(rtg, config.TARGET_RTG_PERCENTILE_LOW))

        ctx_R_high = ctx_R.copy(); ctx_R_high[-1] = rtg_high
        ctx_R_low  = ctx_R.copy(); ctx_R_low[-1]  = rtg_low

        pred_high = model.forward(ctx_R_high[None, :], ctx_S[None, :], ctx_A[None, :])[0, -1]
        pred_low  = model.forward(ctx_R_low[None, :],  ctx_S[None, :], ctx_A[None, :])[0, -1]

        action_signal   = float(pred_high)
        rtg_sensitivity = float(pred_high - pred_low)

        # In-sample edge: correlation between predicted (true-RTG-conditioned)
        # actions and realized forward returns across the training tape
        N_eval = n - K + 1
        R_eval = np.stack([rtg[j:j + K] for j in range(N_eval)])
        S_eval = np.stack([states[j:j + K] for j in range(N_eval)])
        A_eval = np.stack([actions[j:j + K] for j in range(N_eval)])
        pred_eval = model.forward(R_eval, S_eval, A_eval)[:, -1]
        fwd_eval = fwd[K - 1:K - 1 + N_eval]
        if np.std(pred_eval) > 1e-8 and np.std(fwd_eval) > 1e-8:
            insample_edge = float(np.corrcoef(pred_eval, fwd_eval)[0, 1])
        else:
            insample_edge = 0.0

        print(f"    {ticker}: action={action_signal:.4f}  "
              f"sensitivity={rtg_sensitivity:.4f}  edge={insample_edge:.4f}")

        composite = (
            config.WEIGHT_ACTION      * action_signal
            + config.WEIGHT_SENSITIVITY * rtg_sensitivity
            + config.WEIGHT_EDGE         * insample_edge
        )
        raw_scores[ticker] = composite

    if not raw_scores:
        return pd.Series(dtype=float)

    scores = pd.Series(raw_scores)
    mu, std = scores.mean(), scores.std()
    if std < 1e-10:
        return pd.Series(0.0, index=scores.index)
    return (scores - mu) / std
