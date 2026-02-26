# Long-Term vs Short-Term Metrics for Music Recommendation Systems

A simulation-based framework for evaluating how music recommender systems can balance **immediate engagement** (short-term) with **user retention and content diversity** (long-term), built on the [Yandex Music (Yambda)](https://huggingface.co/datasets/yandex/yambda) large-scale listening dataset.

> **Venue**: UMAP 2026 — ACM Conference on User Modeling, Adaptation and Personalization

---

## Table of Contents

- [Objective](#objective)
- [Experiment Design](#experiment-design)
  - [Three-Layer Metric Framework](#three-layer-metric-framework)
  - [Recommendation Policies](#recommendation-policies)
  - [Candidate Expansion & Reranking](#candidate-expansion--reranking)
  - [Stress Testing](#stress-testing)
- [Dataset](#dataset)
- [Notebook Walkthrough](#notebook-walkthrough)
  - [Part 1 — Data Loading & Metric Construction](#part-1--data-loading--metric-construction)
  - [Part 2 — Analysis, Policies & Evaluation](#part-2--analysis-policies--evaluation)
- [Key Outputs & Visualizations](#key-outputs--visualizations)
- [Requirements](#requirements)
- [Reproducing the Results](#reproducing-the-results)
- [License](#license)

---

## Objective

Modern music recommender systems are typically optimized for short-term proxy metrics such as click-through rate or skip rate. While these signals are easy to measure, they can lead to **filter bubbles**, **popularity bias**, and **user churn** over time.

This project proposes and evaluates a **multi-layer satisfaction framework** that jointly considers:

1. **Immediate interaction quality** — did the user enjoy *this* track?
2. **Session-level engagement** — how rich and diverse was the listening session?
3. **Long-term retention** — does the user keep coming back?

By simulating two contrasting recommendation policies on real listening data, the framework quantifies the **trade-off between short-term engagement and long-term health** of a music platform.

---

## Experiment Design

### Three-Layer Metric Framework

| Layer | Granularity | Metrics | Captures |
|-------|-------------|---------|----------|
| **Layer 1** | Per-listen | Skip rate, completion rate, mean played ratio, like rate | Immediate interaction quality |
| **Layer 2** | Per-session | Session length, unique track ratio, algorithmic mix ratio, in-session skip rate | Within-session engagement & diversity |
| **Layer 3** | Per-user | Number of sessions, active span (days), mean return gap (hours), retention signal | Long-term retention & loyalty |

**Session definition**: A new session begins whenever the gap between consecutive listens by the same user exceeds **30 minutes** (360 × 5-second time bins).

**Threshold calibration**: Skip and completion thresholds are derived empirically from the data distribution — skip is defined as playback below the **25th percentile** of `played_ratio`, and completion as playback above the **90th percentile** (or 0.90 if the empirical quantile saturates at 1.0).

### Recommendation Policies

Two simulated policies are compared via **candidate-set reranking**:

| Policy | Scoring Function | Philosophy |
|--------|-----------------|------------|
| **Policy A** *(Short-term)* | `score = played_ratio` | Maximize immediate engagement only |
| **Policy B** *(Multi-layer)* | `score = w₁·played_ratio + w₂·inv_pop + w₃·retention_signal` | Balance engagement, novelty, and retention |

Where:
- **`played_ratio`** — item-level mean playback completion (engagement proxy)
- **`inv_pop`** — `1 / log(1 + popularity)`, an inverse-popularity novelty signal that promotes long-tail content
- **`retention_signal`** — normalized inverse of mean return gap per user (higher = more loyal user); min-max scaled to [0, 1]

Default weights: `w₁ = 0.4`, `w₂ = 0.3`, `w₃ = 0.3`.

### Candidate Expansion & Reranking

For each session, a candidate set is constructed by combining:

1. **Tracks already in the session** (ground-truth interactions)
2. **Up to 50 items sampled from the user's history** (personalization)
3. **Up to 50 items sampled globally** (exploration / cold-start)

Both policies independently rerank the same candidate set and select the **Top-20** items. Session-level summary metrics (predicted engagement, inverse popularity, long-tail share) are then compared across policies.

### Stress Testing

Three weight configurations are evaluated to map the **engagement–exposure trade-off frontier**:

| Configuration | w₁ (engagement) | w₂ (novelty) | w₃ (retention) |
|---------------|-----------------|---------------|-----------------|
| `engagement_heavy` | 0.6 | 0.2 | 0.2 |
| `balanced` | 0.5 | 0.25 | 0.25 |
| `diversity_heavy` | 0.4 | 0.3 | 0.3 |

---

## Dataset

**Yandex Music (Yambda)** — a large-scale, publicly available music listening dataset.

- **Source**: `yandex/yambda` on Hugging Face (`flat-multievent-50m` subset)
- **Sample used**: 1,000,000 rows (stream-loaded)
- **Key fields**: `uid`, `item_id`, `timestamp`, `is_organic`, `event_type`, `played_ratio_pct`, `track_length_seconds`
- **Event types**: `listen`, `like`, and others; the analysis focuses on `listen` events for playback-based metrics

---

## Notebook Walkthrough

The notebook `UMAP_Yambda_Satisfaction_Framework.ipynb` is organized into two major parts.

### Part 1 — Data Loading & Metric Construction

| Cell(s) | Description |
|---------|-------------|
| 1 | Dependency installation (commented out) |
| 2 | Import libraries and define constants (`MAX_ROWS=1M`, `SEED=42`) |
| 3 | Stream-load the Yambda dataset from Hugging Face |
| 4 | Pull 1M rows into a Pandas DataFrame |
| 5 | Select relevant columns and enforce clean dtypes |
| 6 | Sanity checks — event type distribution, null rates for `played_ratio_pct` and `track_length_seconds` |
| 7 | Filter to `listen` events; convert `played_ratio_pct` (1–100) → `played_ratio` (0–1) |
| 8 | **Session construction** — sort by user+timestamp, compute inter-listen gaps, assign session IDs using a 30-minute gap threshold |
| 9 | **Layer 1 metrics** — global skip rate, completion rate, mean played ratio |
| 10 | **Layer 2 metrics** — per-session aggregates: session length, unique track ratio, algorithmic mix, in-session skip rate |
| 11 | Descriptive statistics for session-level metrics |
| 12 | **Layer 3 metrics** — per-user aggregates: number of sessions, active span in days, mean return gap in hours |
| 13 | Export intermediate DataFrames as Parquet files for reproducibility |

### Part 2 — Analysis, Policies & Evaluation

| Cell(s) | Description |
|---------|-------------|
| 14 | Markdown header — Part 2 |
| 15–16 | Data cleaning: drop nulls in `played_ratio`, clip values to [0, 1] |
| 17 | **Visualization**: histogram of `played_ratio` distribution |
| 18 | Quantile analysis of played ratio for threshold calibration |
| 19 | Define calibrated skip and completion thresholds |
| 20 | Recompute **Layer 1 metrics** with calibrated thresholds; compute like rate |
| 21 | **Organic vs algorithmic comparison** — skip and completion rates by source |
| 22 | Rebuild sessions (threshold reminder) |
| 23 | Session-level descriptive stats |
| 24 | Re-clean played ratio (clip + drop nulls) |
| 25–26 | **Quantile-based threshold justification** — derive thresholds from the 25th / 90th percentiles with a methodological sentence for the paper |
| 27 | Generate a plain-English methods sentence for thresholds |
| 28 | **Layer 1 summary** (global + organic vs algorithmic breakdown) |
| 29 | **Layer 2 summary** — session-level descriptive statistics |
| 30 | **Layer 3 summary** — user-level descriptive statistics |
| 31 | **Retention signal construction** — inverse of mean return gap, min-max normalized to [0, 1] |
| 32 | Item popularity computation |
| 33 | Merge retention signal into listens DataFrame |
| 34 | Inspect merged columns |
| 35 | Compute inverse-popularity (`inv_pop`) novelty signal |
| 36 | **Policy scoring** — compute `score_A` (short-term) and `score_B` (multi-layer) for every listen |
| 37 | **Candidate expansion reranking** — for each session, build candidate set (session + user history + global), rerank under both policies, extract Top-20 summary metrics |
| 38 | **Stress testing** — sweep three weight configurations, compute engagement and novelty deltas |
| 39 | **Visualization**: engagement–exposure trade-off frontier scatter plot |
| 40 | Policy A vs Policy B — mean Top-k metrics comparison |
| 41 | Per-session delta analysis (B − A) across Top-k metrics |
| 42 | **Visualization**: played ratio distribution histogram |
| 43 | **Visualization**: organic vs algorithmic bar chart (skip rate + completion rate) |
| 44 | Per-session delta with robust column filtering |
| 45 | **Visualization**: Policy B − Policy A bar chart for Top-k session metrics |
| 46 | **Correlation matrix** — played ratio, inverse popularity, retention signal, and composite score B |

---

## Key Outputs & Visualizations

| Output | Description |
|--------|-------------|
| `yambda_sample_flat50m.parquet` | Raw 1M-row sample |
| `yambda_sample_listens.parquet` | Cleaned listen events with sessions |
| `yambda_session_stats.parquet` | Layer 2 session-level aggregates |
| `yambda_user_stats.parquet` | Layer 3 user-level aggregates |
| Played Ratio Histogram | Shows bimodal distribution — many skips (near 0) and many completions (near 1) |
| Organic vs Algorithmic Bar Chart | Compares skip and completion rates across recommendation sources |
| Engagement–Exposure Trade-off Frontier | Scatter plot showing how different weight configs shift the novelty-engagement balance |
| Policy B − A Delta Bar Chart | Quantifies per-metric lift of the multi-layer policy over the engagement-only baseline |
| Correlation Matrix | Reveals relationships between engagement, novelty, retention, and the composite score |

---

## Requirements

- Python ≥ 3.9
- `pandas`
- `numpy`
- `matplotlib`
- `datasets` (Hugging Face)
- `tqdm`
- `pyarrow`

Install all dependencies:

```bash
pip install pandas numpy matplotlib datasets tqdm pyarrow
```

---

## Reproducing the Results

```bash
# Clone the repository
git clone https://github.com/hemanth-rj/long-term-vs-short-term-metrics-music-recsys.git
cd long-term-vs-short-term-metrics-music-recsys

# Install dependencies
pip install pandas numpy matplotlib datasets tqdm pyarrow

# Run the notebook
jupyter notebook UMAP_Yambda_Satisfaction_Framework.ipynb
```

> **Note**: The first run will stream ~1M rows from the Yambda dataset on Hugging Face, which requires an internet connection. Subsequent runs can load from the saved Parquet files.

---

## License

See repository for license details.
