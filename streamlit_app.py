import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Decision Transformer Engine", layout="wide")

st.markdown("""
<style>
.main-header { font-size:2.4rem; font-weight:700; color:#0d1b2a; margin-bottom:0.3rem; }
.sub-header  { font-size:1.1rem; color:#555; margin-bottom:1.5rem; }
.uni-title   { font-size:1.4rem; font-weight:600; margin-top:1rem; margin-bottom:0.8rem;
               padding-left:0.5rem; border-left:5px solid #1b4965; }
.etf-card    { background:linear-gradient(135deg,#0d1b2a 0%,#1b4965 100%); color:white;
               border-radius:14px; padding:1rem; margin:0.4rem; text-align:center;
               box-shadow:0 4px 6px rgba(0,0,0,0.2); }
.win-card    { background:linear-gradient(135deg,#0d1b2a 0%,#274156 100%); color:white;
               border-radius:14px; padding:1rem; margin:0.4rem; text-align:center;
               box-shadow:0 4px 6px rgba(0,0,0,0.2); }
.etf-ticker  { font-size:1.3rem; font-weight:bold; }
.etf-score   { font-size:0.88rem; margin-top:0.25rem; opacity:0.9; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎯 Decision Transformer Engine</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Chen et al. (2021) Decision Transformer · '
    'Offline RL as causal sequence modelling on (Return-to-go, State, Action) · '
    'Return-conditioned policy, trained by behavior cloning · '
    'Multi-window cross-sectional z-score</div>',
    unsafe_allow_html=True)

st.sidebar.markdown("## Decision Transformer Engine")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown(f"**Windows:** {config.WINDOWS}")
st.sidebar.markdown(
    f"**Context:** {config.CONTEXT_LEN} steps | "
    f"{config.N_LAYERS} layers | {config.N_HEADS} heads | dim={config.EMBED_DIM}")
st.sidebar.markdown(
    f"**Training:** epochs={config.DT_EPOCHS} | lr={config.DT_LR} | "
    f"batch={config.DT_BATCH_SIZE}")
st.sidebar.markdown(
    f"**Target RTG:** high=p{config.TARGET_RTG_PERCENTILE_HIGH} | "
    f"low=p{config.TARGET_RTG_PERCENTILE_LOW}")
st.sidebar.markdown(
    f"**Weights:** Action {config.WEIGHT_ACTION:.0%} | "
    f"Sensitivity {config.WEIGHT_SENSITIVITY:.0%} | "
    f"Edge {config.WEIGHT_EDGE:.0%}")

HF_TOKEN    = config.HF_TOKEN
OUTPUT_REPO = config.OUTPUT_REPO


@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        return [f["name"] for f in fs.ls(f"datasets/{OUTPUT_REPO}",
                                          detail=True, recursive=True)
                if f["type"] == "file"]
    except Exception as e:
        return [f"Error: {e}"]


def find_latest(files, prefix):
    matches = sorted([f for f in files if f.endswith(".json") and prefix in f],
                     reverse=True)
    return matches[0] if matches else None


@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


files     = list_repo_files()
tab1_path = find_latest(files, "dt_engine_2")
tab2_path = find_latest(files, "dt_engine_windows_")

if not tab1_path:
    st.error("No results found. Run trainer.py first.")
    st.stop()

data1 = load_json(tab1_path)
if "error" in data1:
    st.error(f"Error loading data: {data1['error']}")
    st.stop()

data2      = load_json(tab2_path) if tab2_path else None
universes1 = data1["universes"]
universes2 = data2["universes"] if data2 and "error" not in data2 else None

st.sidebar.markdown(f"**Run date:** `{data1.get('run_date','?')}`")

tab1, tab2 = st.tabs(["🏆 Best Window per ETF", "🔍 Explore by Window"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🏆 Top ETFs — Return-Conditioned Policy Signal")

    with st.expander("Decision Transformer Methodology", expanded=True):
        st.markdown("""
Offline RL reframed as **causal sequence modelling** on a fixed historical tape —
no environment interaction required (unlike DQN-ENGINE / ERL-ENGINE / CAUSAL-RL,
all online RL in this suite):

```
trajectory = (R_1, s_1, a_1, R_2, s_2, a_2, ..., R_K, s_K, a_K)
```

- **R_t** — return-to-go: sum of future rewards from t onward
- **s_t** — market state: normalized lagged returns + macro context
- **a_t** — action: position taken, in [-1, +1]

**Offline dataset construction** (hindsight-optimal, from realized returns):

```
a_t = tanh(ACTION_SCALE * forward_return_t)      (behavior-cloning target)
r_t = a_t * forward_return_t                       (reward achieved)
R_t = sum_{k=t}^{T} r_k                             (return-to-go, undiscounted)
```

A causal (GPT-style) transformer — built from scratch, analytical backprop,
no autograd — is trained by ordinary behavior cloning: predict a_t from the
causally-masked context up to and including s_t.

**Conditioning on desired return at inference:**

```
"What action would a trajectory achieving a HIGH return-to-go take here?"
```

Feed in an ambitious R_t (90th percentile of the in-sample return-to-go
distribution) and read off the model's recommended action — the return-
conditioned policy.

**Signal:**

```
score = 0.50 * action_signal + 0.25 * rtg_sensitivity + 0.25 * insample_edge
```

- `action_signal` — predicted position when conditioned on a HIGH desired return
- `rtg_sensitivity` — action_signal minus the action predicted at a LOW desired
  return; measures how responsive the policy actually is to what it's asked for
- `insample_edge` — correlation between predicted actions and realized forward
  returns across the training window; validates the policy tracks genuinely
  profitable behavior

**Distinct from DQN-ENGINE / ERL-ENGINE / CAUSAL-RL:** those learn through
online interaction (TD error, evolutionary fitness, causal effect estimation)
and must sample the environment repeatedly. The Decision Transformer only
ever sees the fixed historical tape once — pure offline behavior cloning,
avoiding the online sample-efficiency problem entirely.
        """)

    for universe_name, uni_data in universes1.items():
        top_etfs = uni_data.get("top_etfs", [])
        if not top_etfs:
            continue
        st.markdown(
            f'<div class="uni-title">{universe_name.replace("_"," ").title()}</div>',
            unsafe_allow_html=True)
        cols = st.columns(3)
        for idx, etf in enumerate(top_etfs):
            with cols[idx]:
                st.markdown(f"""
<div class="etf-card">
  <div class="etf-ticker">{etf['ticker']}</div>
  <div class="etf-score">DT score = {etf['dt_score']:.4f}</div>
  <div class="etf-score">best window = {etf.get('best_window','N/A')}d</div>
</div>
""", unsafe_allow_html=True)

        with st.expander(f"Full ranking — {universe_name}"):
            full = uni_data.get("full_scores", {})
            if full:
                rows = []
                for t, info in full.items():
                    score = info.get("score", info) if isinstance(info, dict) else info
                    win   = info.get("best_window", "N/A") if isinstance(info, dict) else "N/A"
                    rows.append({"ETF": t, "DT Score": score, "Best Window (d)": win})
                df = pd.DataFrame(rows).sort_values("DT Score", ascending=False)
                st.dataframe(df, use_container_width=True, hide_index=True)
        st.divider()

    st.caption(
        f"Run date: {data1.get('run_date','?')} · "
        "Chen et al. (2021) Decision Transformer · "
        "Scores are cross-sectional z-scores.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🔍 Explore Decision Transformer Rankings by Window")

    if not universes2:
        st.warning("Window-level detail not found. Re-run trainer.")
        st.stop()

    all_wins = set()
    for ud in universes2.values():
        all_wins.update(ud.get("windows", {}).keys())
    win_options = sorted([int(w) for w in all_wins])

    if not win_options:
        st.error("No window data available.")
        st.stop()

    default_idx  = win_options.index(252) if 252 in win_options else 0
    selected_win = st.selectbox(
        "Select lookback window",
        options=win_options,
        index=default_idx,
        format_func=lambda w: f"{w}d  (~{round(w/21)} months)",
    )
    win_key = str(selected_win)

    with st.expander("Window guidance", expanded=False):
        st.markdown("""
- **63d** — short offline tape; policy trained on the recent regime; reactive
- **126d** — 6-month tape; recommended minimum for a stable policy
- **252d** — 1-year tape; most stable return-conditioned policy; recommended primary signal
- **504d** — 2-year tape; structural regime policy; slow-moving signal
        """)

    st.markdown(f"### Decision Transformer Rankings at **{selected_win}d** window")

    for universe_name in ["FI_COMMODITIES", "EQUITY_SECTORS", "COMBINED"]:
        label = {
            "FI_COMMODITIES": "🏦 FI & Commodities",
            "EQUITY_SECTORS": "📈 Equity Sectors",
            "COMBINED":       "🌐 Combined",
        }.get(universe_name, universe_name)

        st.markdown(f'<div class="uni-title">{label}</div>', unsafe_allow_html=True)

        uni_data = universes2.get(universe_name, {})
        win_data = uni_data.get("windows", {}).get(win_key)

        if not win_data:
            st.info(f"No data for {universe_name} at {selected_win}d.")
            st.divider()
            continue

        cols = st.columns(3)
        for idx, etf in enumerate(win_data.get("top_etfs", [])):
            with cols[idx]:
                st.markdown(f"""
<div class="win-card">
  <div class="etf-ticker">{etf['ticker']}</div>
  <div class="etf-score">DT score = {etf['dt_score']:.4f}</div>
  <div class="etf-score">window = {selected_win}d</div>
</div>
""", unsafe_allow_html=True)

        with st.expander(f"Full ranking — {label} @ {selected_win}d"):
            rows = win_data.get("full_ranking", [])
            if rows:
                df = pd.DataFrame(rows, columns=["ETF", "DT Score"])
                df.insert(0, "Rank", range(1, len(df) + 1))
                st.dataframe(df, use_container_width=True, hide_index=True)

        st.divider()

    st.caption(f"Window: {selected_win}d · Run date: {data2.get('run_date','?')}")
