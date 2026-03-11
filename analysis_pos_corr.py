"""POS Probability Correlation: Algorithms 2 & 3.

Algorithm-2: Per-POS Pearson/Spearman correlation across 144 contexts.
Algorithm-3: Delta-table (cell-wise absolute difference) approach.
"""

import pathlib

import numpy as np
import pandas as pd
from scipy import stats

from filter_data import load_gpt, load_human

OUT = pathlib.Path("output/pos_corr")
OUT.mkdir(parents=True, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────
human_raw = load_human()
model_raw = load_gpt()

# ── Context lookup ───────────────────────────────────────────────────
context_lookup = (
    human_raw.drop_duplicates("word.id")[["word.id", "Left context"]]
    .rename(columns={"word.id": "target_word", "Left context": "left_context"})
)

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
corr_df.to_csv(OUT / "per_pos_correlation.csv", index=False)

print("=== Algorithm-2: Per-POS correlation ===")
print(corr_df.to_string(index=False))
print()

# ── Save POS matrices with context columns ───────────────────────────
def _add_context_cols(pos_df):
    """Reset index, rename to target_word, merge left_context, reorder."""
    df = pos_df.reset_index().rename(columns={"word_id": "target_word"})
    df = df.merge(context_lookup, on="target_word", how="left")
    cols = ["left_context", "target_word"] + [c for c in df.columns if c not in ("left_context", "target_word")]
    return df[cols]

_add_context_cols(human_pos).to_csv(OUT / "human_pos_matrix.csv", index=False)
_add_context_cols(model_pos).to_csv(OUT / "model_pos_matrix.csv", index=False)

# ── Algorithm-3: Delta-table ─────────────────────────────────────────
delta = (human_pos - model_pos).abs()
mean_delta_per_context = delta.mean(axis=1)

delta_out = pd.DataFrame({"target_word": all_contexts, "mean_delta": mean_delta_per_context.values})
for pos in all_pos:
    delta_out[pos] = delta[pos].values
delta_out = delta_out.merge(context_lookup, on="target_word", how="left")
cols = ["left_context", "target_word"] + [c for c in delta_out.columns if c not in ("left_context", "target_word")]
delta_out = delta_out[cols]
delta_out.to_csv(OUT / "per_context_delta.csv", index=False)

overall_mean_delta = mean_delta_per_context.mean()

# ── Summary ──────────────────────────────────────────────────────────
summary = pd.DataFrame({"overall_mean_delta": [round(overall_mean_delta, 4)]})
summary.to_csv(OUT / "summary.csv", index=False)

print("=== Algorithm-3: Delta-table ===")
print(f"Overall mean delta: {overall_mean_delta:.4f}")
print(f"Per-context mean delta: min={mean_delta_per_context.min():.4f}, "
      f"max={mean_delta_per_context.max():.4f}")
print(f"\nOutputs saved to {OUT}/")
