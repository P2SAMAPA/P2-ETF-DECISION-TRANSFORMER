import os

HF_TOKEN    = os.environ.get("HF_TOKEN", "")
DATA_REPO   = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-decision-transformer-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "SMH", "SOXX", "XLB",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "SMH", "SOXX", "XLB",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
}

MACRO_COLS_CORE     = ["VIX", "DXY", "T10Y2Y"]
MACRO_COLS_EXTENDED = ["IG_SPREAD", "HY_SPREAD"]

# ── Rolling windows (trading days) ────────────────────────────────────────────
WINDOWS = [63, 126, 252, 504]

# ── Decision Transformer hyperparameters ─────────────────────────────────────
# Chen et al. (2021) "Decision Transformer: Reinforcement Learning via
# Sequence Modeling". Offline RL is reframed as causal sequence modelling on
# (Return-to-go, State, Action) triples:
#
#   trajectory = (R_1, s_1, a_1, R_2, s_2, a_2, ..., R_K, s_K, a_K)
#
# where R_t = sum of future rewards from t onward (return-to-go), s_t is the
# market state, and a_t is the position taken. Trained via behavior cloning
# (predict a_t from the causal context) on hindsight-optimal offline
# trajectories built from realized historical data — no online interaction,
# no environment rollout, no replay buffer. This is what makes it "offline":
# unlike DQN-ENGINE / ERL-ENGINE / CAUSAL-RL in the suite (all online RL that
# must interact with — or simulate interacting with — an environment to
# collect experience), the Decision Transformer only ever sees the fixed
# historical tape once, avoiding the online sample-efficiency problem entirely.

CONTEXT_LEN = 20     # K: number of past (R,s,a) triples in the context window
N_LAGS      = 10     # lagged return features included in the state s_t

# Transformer architecture (causal self-attention, built from scratch,
# analytical backprop — no autograd framework)
EMBED_DIM = 24
N_HEADS   = 4
N_LAYERS  = 2
FF_HIDDEN = 48

# Hindsight-optimal action used as the behavior-cloning target:
#   a_t = tanh(ACTION_SCALE * forward_return_t)
# This constructs the offline (state, action, reward) tape from realized
# returns — the "expert demonstrations" a true offline RL dataset would need.
ACTION_SCALE = 40.0

# Reward for a given step: r_t = a_t * forward_return_t (position * realized
# return achieved by that position). GAMMA=1.0: no discounting within window.
GAMMA = 1.0

# Forward return horizon defining each step's reward
PRED_HORIZON = 21

# At inference we condition on a desired HIGH return-to-go — "what would a
# high-return trajectory do here?" — set to this percentile of the in-sample
# realized return-to-go distribution. A LOW percentile is also evaluated to
# measure how responsive the learned policy is to the return it's asked for.
TARGET_RTG_PERCENTILE_HIGH = 90
TARGET_RTG_PERCENTILE_LOW  = 10

# Training
DT_EPOCHS     = 50
DT_LR         = 3e-3
DT_BATCH_SIZE = 16

# ── Score construction ────────────────────────────────────────────────────────
# action_signal : predicted action (position) when conditioned on the desired
#                 HIGH return-to-go — direction & magnitude of the recommended
#                 trade a high-performing trajectory would take today
# rtg_sensitivity: action_signal minus the action predicted when conditioned
#                 on a LOW return-to-go. A well-trained policy should recommend
#                 a more aggressive long when asked for a higher return; this
#                 measures how responsive (and therefore how meaningful) the
#                 policy's return-conditioning actually is
# insample_edge : correlation between the model's predicted actions and the
#                 realized forward returns across the training window —
#                 validates that the learned policy tracks genuinely
#                 profitable behavior rather than just fitting noise

WEIGHT_ACTION      = 0.50
WEIGHT_SENSITIVITY = 0.25
WEIGHT_EDGE        = 0.25

TOP_N = 3
