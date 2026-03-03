"""POS Probability Correlation: Algorithms 2 & 3.

Algorithm-2: Per-POS Pearson/Spearman correlation across 144 contexts.
Algorithm-3: Delta-table (cell-wise absolute difference) approach.
"""

import pathlib

import numpy as np
import pandas as pd
from scipy import stats

OUT = pathlib.Path("output")
OUT.mkdir(exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────
human_raw = pd.read_csv("people_with_prob.csv")
model_raw = pd.read_csv("gpt4omini_morph_2.csv")

# ── Deduplicate ──────────────────────────────────────────────────────
# Human: one row per (context, answer) — take first row's probability_y & upos
human = (
    human_raw.groupby(["word.id", "answer"])
    .first()
    .reset_index()[["word.id", "answer", "upos_answer", "probability_y"]]
)

# Model: one row per (context, prediction_cleaned) — keep highest probability
model = (
    model_raw.dropna(subset=["upos_word"])
    .sort_values("probability_converted", ascending=False)
    .drop_duplicates(subset=["target_word_id", "prediction_cleaned"], keep="first")[
        ["target_word_id", "prediction_cleaned", "upos_word", "probability_converted"]
    ]
)

# ── Build POS probability tables ─────────────────────────────────────
human_pos = (
    human.groupby(["word.id", "upos_answer"])["probability_y"]
    .sum()
    .unstack(fill_value=0)
)
human_pos.index.name = "word_id"

model_pos = (
    model.groupby(["target_word_id", "upos_word"])["probability_converted"]
    .sum()
    .unstack(fill_value=0)
)
model_pos.index.name = "word_id"

# Align columns (union of all POS tags) and rows (144 contexts)
all_pos = sorted(set(human_pos.columns) | set(model_pos.columns))
all_contexts = sorted(set(human_pos.index) | set(model_pos.index))

human_pos = human_pos.reindex(index=all_contexts, columns=all_pos, fill_value=0)
model_pos = model_pos.reindex(index=all_contexts, columns=all_pos, fill_value=0)

print(f"POS tags: {len(all_pos)}  Contexts: {len(all_contexts)}")
print(f"POS tags: {all_pos}\n")

# ── Algorithm-2: Per-POS correlation across contexts ─────────────────
MIN_CONTEXTS = 10
rows = []
for pos in all_pos:
    h_vec = human_pos[pos].values
    m_vec = model_pos[pos].values
    # Count contexts where at least one source has non-zero probability
    n_ctx = int(np.sum((h_vec > 0) | (m_vec > 0)))
    if n_ctx < MIN_CONTEXTS:
        continue
    # Skip if either vector is constant (correlation undefined)
    if np.std(h_vec) == 0 or np.std(m_vec) == 0:
        continue
    pr, pp = stats.pearsonr(h_vec, m_vec)
    sr, sp = stats.spearmanr(h_vec, m_vec)
    rows.append(
        {
            "pos_tag": pos,
            "n_contexts": n_ctx,
            "pearson_r": round(pr, 4),
            "pearson_p": pp,
            "spearman_r": round(sr, 4),
            "spearman_p": sp,
        }
    )

corr_df = pd.DataFrame(rows).sort_values("pearson_r", ascending=False)
corr_df.to_csv(OUT / "pos_correlation.csv", index=False)

print("=== Algorithm-2: Per-POS correlation ===")
print(corr_df.to_string(index=False))
print()

# ── Algorithm-3: Delta-table ─────────────────────────────────────────
delta = (human_pos - model_pos).abs()
mean_delta_per_context = delta.mean(axis=1)

delta_out = pd.DataFrame({"word_id": all_contexts, "mean_delta": mean_delta_per_context.values})
for pos in all_pos:
    delta_out[pos] = delta[pos].values
delta_out.to_csv(OUT / "pos_delta.csv", index=False)

overall_mean_delta = mean_delta_per_context.mean()
print("=== Algorithm-3: Delta-table ===")
print(f"Overall mean delta: {overall_mean_delta:.4f}")
print(f"Per-context mean delta: min={mean_delta_per_context.min():.4f}, "
      f"max={mean_delta_per_context.max():.4f}")
print(f"\nDelta saved to {OUT / 'pos_delta.csv'}")
print(f"Correlation saved to {OUT / 'pos_correlation.csv'}")
