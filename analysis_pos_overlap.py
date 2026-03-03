"""POS Distribution Overlap & Algorithm-1 (ranked POS intersection@k).

Compares the probability-weighted POS distributions of human cloze responses
and GPT-4o-mini predictions across 144 Russian sentence contexts.

Outputs: output/pos_overlap.csv  (per-context results)
"""

import pathlib

import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

# ── load data ──────────────────────────────────────────────────────────
human = pd.read_csv(ROOT / "people_with_prob.csv")
gpt = pd.read_csv(ROOT / "gpt4omini_morph_2.csv")

# ── human POS distribution ─────────────────────────────────────────────
# Dedup: one row per (context, answer) — take first probability_y & upos
human_dedup = (
    human.groupby(["word.id", "answer"], as_index=False)
    .first()[["word.id", "answer", "upos_answer", "probability_y"]]
)
# Sum probabilities by (context, POS)
human_pos = (
    human_dedup.groupby(["word.id", "upos_answer"], as_index=False)["probability_y"]
    .sum()
    .rename(columns={"word.id": "word_id", "upos_answer": "pos", "probability_y": "weight"})
)

# ── model POS distribution ─────────────────────────────────────────────
# Dedup: one row per (context, prediction_cleaned) — keep highest prob
gpt_sorted = gpt.sort_values("probability_converted", ascending=False)
gpt_dedup = gpt_sorted.drop_duplicates(subset=["target_word_id", "prediction_cleaned"], keep="first")
gpt_dedup = gpt_dedup[["target_word_id", "prediction_cleaned", "upos_word", "probability_converted"]]

# Sum probabilities by (context, POS)
model_pos = (
    gpt_dedup.groupby(["target_word_id", "upos_word"], as_index=False)["probability_converted"]
    .sum()
    .rename(columns={"target_word_id": "word_id", "upos_word": "pos", "probability_converted": "weight"})
)

# ── per-context rankings ───────────────────────────────────────────────
def ranked_pos(pos_df: pd.DataFrame) -> dict[str, list[str]]:
    """Return {word_id: [pos_tag sorted by descending weight]}."""
    out = {}
    for wid, grp in pos_df.groupby("word_id"):
        ordered = grp.sort_values("weight", ascending=False)["pos"].tolist()
        out[wid] = ordered
    return out


human_ranked = ranked_pos(human_pos)
model_ranked = ranked_pos(model_pos)

all_contexts = sorted(set(human_ranked) & set(model_ranked))
print(f"Contexts with both human and model data: {len(all_contexts)}")

# ── compute metrics ────────────────────────────────────────────────────
K_VALUES = [1, 2, 3, 4, 5]
rows = []

for ctx in all_contexts:
    h_rank = human_ranked[ctx]
    m_rank = model_ranked[ctx]

    human_top = h_rank[0]
    model_top = m_rank[0]
    match_at_1 = int(human_top == model_top)

    intersections = {}
    for k in K_VALUES:
        h_set = set(h_rank[:k])
        m_set = set(m_rank[:k])
        intersections[k] = len(h_set & m_set) / k

    rows.append(
        {
            "word_id": ctx,
            "human_top_pos": human_top,
            "model_top_pos": model_top,
            "pos_match_at_1": match_at_1,
            **{f"pos_intersection_at_{k}": intersections[k] for k in K_VALUES},
        }
    )

results = pd.DataFrame(rows)
results.to_csv(OUT / "pos_overlap.csv", index=False)
print(f"Saved {len(results)} rows to {OUT / 'pos_overlap.csv'}")

# ── summary statistics ─────────────────────────────────────────────────
print("\n=== Mean values across all contexts ===")
print(f"  POS overlap@1 (exact top-POS match): {results['pos_match_at_1'].mean():.4f}")
for k in K_VALUES:
    col = f"pos_intersection_at_{k}"
    print(f"  POS intersection@{k}: {results[col].mean():.4f}")
