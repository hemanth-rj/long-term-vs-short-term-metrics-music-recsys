# Offline Ranking Objective Audit for Music Streaming

**UMAP 2026 Industry Track — Companion Code**

A lightweight framework for auditing the engagement-diversity trade-off of alternative ranking objectives in music streaming, using candidate re-ranking on historical listening logs — without any model retraining.

---

## Overview

Production recommender systems in music streaming typically optimise for short-term engagement: maximising the fraction of a track a user plays before skipping. This concentrates recommendations on popular content and systematically suppresses the long tail of the catalogue.

Evaluating a diversity-aware alternative normally requires full model retraining and live A/B experimentation — a process that takes weeks and carries real production risk.

This framework answers the question **before** any system change is made:

> *If we changed our ranking objective to incorporate novelty and retention signals, how much engagement would we give up and how much long-tail exposure would we gain?*

The framework simulates two ranking policies on held-out sessions from historical data, using popularity-calibrated candidate sampling to ensure the baseline reflects realistic production conditions.

---

## Key Results (Yambda, 1M events, alpha=1.0)

| Metric | Policy A (engagement only) | Policy B (multi-layer) | Delta |
|---|---|---|---|
| Predicted engagement | 0.9297 | 0.9154 | **−0.0143** |
| Long-tail share | 0.2939 | 0.3629 | **+0.0690** |
| Inv. popularity | 0.6582 | 0.7437 | **+0.0856** |
| Retention affinity | 0.0975 | 0.0987 | **+0.0012** |

All differences significant at p < 0.001 across 22,970 evaluation sessions.

**Stress test (balanced config, alpha=1.0):** +9.5pp long-tail gain at −2.6pp engagement cost. Trade-off is strictly monotonic across all three weight configurations tested.

---

## Dataset

[Yambda](https://huggingface.co/datasets/yandex/yambda) — a large-scale public music streaming dataset by Yandex.

- **Subset used:** `flat-multievent-50m`
- **Events sampled:** 1,000,000
- **Listen events:** 971,902
- **Unique items (train window):** 93,778
- **Users:** 194
- **Evaluation sessions:** 22,970

The dataset is loaded via HuggingFace `datasets` in streaming mode. No manual download required.

---

## Installation

```bash
pip install datasets pyarrow pandas numpy tqdm scipy matplotlib
```

Or using the notebook's commented install cell:

```bash
# !pip -q install datasets pyarrow pandas numpy tqdm scipy
```

**Python version:** 3.12 (tested). Should work on 3.9+.

---

## Notebook Structure

The notebook is organised into five parts. Run cells sequentially from top to bottom.

### Part 1 — Data Loading & Characterisation (Cells 1–16)

Loads 1M events from Yambda, filters to listen events, cleans the played ratio, builds sessions using a 30-minute inactivity threshold, and computes three layers of engagement metrics:

- **Layer 1:** Individual listen metrics — skip rate, completion rate, mean played ratio
- **Layer 2:** Session-level metrics — session length, unique track ratio, algorithmic vs organic ratio
- **Layer 3:** User-level metrics — number of sessions, active span, mean return gap between sessions

Saves four intermediate parquet files:

```
yambda_sample_flat50m.parquet
yambda_sample_listens.parquet
yambda_session_stats.parquet
yambda_user_stats.parquet
```

### Part 2 — Temporal Split & Signal Construction (Cells 17–23)

**Temporal split (80/20):** Item statistics are computed from the first 80% of the timeline only. Evaluation happens on sessions in the remaining 20%. This prevents the circular evaluation problem where Policy A ranks by the exact statistic being measured.

Three item-level scoring signals are computed from the training window:

| Signal | Formula | What it measures |
|---|---|---|
| `item_mean_played_ratio` | Mean played ratio per item (train) | Predicted engagement |
| `inv_pop` | `1 / log(1 + popularity)` | Novelty — rare items score high |
| `item_retention_affinity` | Mean user retention signal per item | Items liked by loyal users score high |

Inter-signal correlations are all below 0.16, confirming each signal captures independent information.

### Part 3 — Popularity-Biased Candidate Sampling (Cells 24–28)

**The core methodological contribution.**

Global candidates are sampled proportional to `pop^alpha` rather than uniformly:

```
P(select item i) = pop_i^alpha / sum_j(pop_j^alpha)
```

| Alpha | Regime | Policy A long-tail baseline |
|---|---|---|
| 0.0 | Uniform (broken — artificially inflates baseline to 68%) | 68.3% |
| 0.5 | Mild popularity bias | 47.2% |
| **1.0** | **Linear popularity weighting (recommended)** | **29.4%** |
| 2.0 | Quadratic concentration | 22.1% |

**Why this matters:** Uniform sampling floods every candidate set with long-tail items by default (median item popularity = 2 listens). This makes Policy A look artificially diverse before any diversity objective is applied, making the problem invisible in the data. Popularity-biased sampling creates a realistic baseline where the engagement-only policy concentrates on popular content — which is what a real production system does.

**Why alpha=1.0:** Linear popularity weighting has a natural interpretation (items sampled at the rate they appear in the listening distribution), requires no exponent justification, and produces a Policy A baseline consistent with prior observations that engagement-optimised systems concentrate exposure on popular content [Abdollahpouri et al., 2019; Steck, 2018].

### Part 4 — Main Experiment: All Alpha Values (Cells 29–30)

Runs the full Policy A vs Policy B comparison for all four alpha values. For each evaluation session:

1. Candidate set is constructed: session items + up to 50 prior user history items (train window only, strictly before session start) + 50 popularity-biased global items
2. Both policies score all candidates
3. Top-20 items are selected under each policy
4. Session-level metrics are recorded

**Policy A** scores by `item_mean_played_ratio` only.

**Policy B** uses adaptive weights:

```
score_B(i, u) = w1(u) * engagement + w2(u) * inv_pop + w3 * retention_affinity

w1(u) = 0.6 - 0.2 * rho_u    (0.6 for low-retention users, 0.4 for high-retention)
w2(u) = 0.2 + 0.2 * rho_u    (0.2 for low-retention users, 0.4 for high-retention)
w3    = 0.2                    (fixed)
```

High-retention users receive more novelty weight; low-retention users remain engagement-heavy.

> **Performance note:** The original implementation using `item_stats[item_stats["item_id"].isin(all_cands)]` inside the session loop took ~9.5 hours per alpha. The fast version in `fast_experiment_cell.py` pre-indexes item stats as numpy arrays and reduces this to ~2–5 minutes per alpha. Use the fast version.

### Part 5 — Results Analysis (Cells 31–41)

Produces the full results suite:

- **Summary table:** Mean metrics per policy, per alpha
- **Delta table:** Policy B minus Policy A, per alpha
- **Paired t-tests:** Statistical significance for all key metrics at all alpha values
- **Figure 1:** Policy A and B long-tail share by alpha (baseline calibration plot)
- **Figure 2:** Absolute engagement by alpha and engagement cost (B−A)
- **Stress test:** Three weight configurations (engagement-heavy / balanced / diversity-heavy) at the best alpha
- **Figure 3:** Trade-off frontier — each weight config as a point in (Δ long-tail, Δ engagement) space
- **Figure 4:** Policy B−A bar chart for the best alpha
- **Final summary:** Paper-ready numbers printout

---

## Key Design Decisions

### Session Construction

Sessions are defined by a 30-minute inactivity gap (360 × 5-second timestamp bins). A new session starts when the gap between consecutive events for a user exceeds this threshold. This is consistent with standard practice in session-based recommendation research.

### Leakage Prevention

User history for candidate generation is restricted to training window events that occurred strictly before the session start timestamp:

```python
prior = [item_id for ts, item_id in history if ts < before_ts]
```

This prevents future information from leaking into the candidate pool — simulating what would actually be available to a production system at ranking time.

### Evaluation Protocol

The evaluation is **paired**: for each session, metrics are computed under both Policy A and Policy B over the same candidate set. Paired t-tests compare per-session deltas to zero, controlling for variation in session composition.

### Long-Tail Definition

Items are defined as long-tail if their training window popularity is at or below the median popularity across all items. With median popularity = 2 listens across 93,778 items, this is a conservative definition that captures the vast majority of the catalogue.

---

## Configurable Parameters

All key parameters are defined at the top of their respective cells and can be modified:

| Parameter | Default | Description |
|---|---|---|
| `MAX_ROWS` | 1,000,000 | Events to load from Yambda |
| `SESSION_GAP_BINS` | 360 | Inactivity threshold for session split (× 5s bins) |
| `SKIP_THRESH` | Q0.25 of played_ratio | Skip definition threshold |
| `COMP_THRESH` | 0.90 | Completion definition threshold |
| `ALPHA_VALUES` | [0.0, 0.5, 1.0, 2.0] | Candidate sampling bias values to test |
| `BEST_ALPHA` | 1.0 | Alpha used for stress test and primary results |
| `TOPK` | 20 | Number of items in top-k recommendation list |
| `N_USER_CANDS` | 50 | User history candidates per session |
| `N_GLOBAL_CANDS` | 50 | Global popularity-biased candidates per session |
| `SEED` | 42 | Random seed for reproducibility |

---

## Output Files

| File | Description |
|---|---|
| `yambda_sample_flat50m.parquet` | Full 1M event sample |
| `yambda_sample_listens.parquet` | Filtered listen events with session IDs |
| `yambda_session_stats.parquet` | Session-level engagement metrics |
| `yambda_user_stats.parquet` | User-level retention metrics |
6. Steck, H. (2018). Calibrated recommendations. *Proceedings of the 12th ACM Conference on Recommender Systems (RecSys 2018).*

7. Yandex (2024). Yambda: Yet another music benchmark dataset for recommendation. *HuggingFace Datasets.* https://huggingface.co/datasets/yandex/yambda
