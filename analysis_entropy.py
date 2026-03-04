import numpy as np
import pandas as pd
from pathlib import Path

K_VALUES = [5, 10, 20, 50, 100]
OUT = Path("output/entropy")
OUT.mkdir(parents=True, exist_ok=True)

people = pd.read_csv("people_with_prob.csv")
gpt = pd.read_csv("gpt4omini_morph_2.csv")

# ── Context lookup table ─────────────────────────────────────────────────
context_lookup = (
    people.drop_duplicates("word.id")[["word.id", "Left context"]]
    .rename(columns={"word.id": "target_word", "Left context": "left_context"})
)

# ── Human entropy per context ────────────────────────────────────────────
# Dedup human answers: groupby word.id + answer, take first probability_y
human_deduped = (
    people.assign(answer_stripped=people["answer"].str.strip())
    .drop_duplicates(subset=["word.id", "answer_stripped"])
)


def shannon_entropy(probs):
    """H = -sum(p_i * log(p_i)) for p_i > 0, using natural log."""
    p = probs[probs > 0]
    return -np.sum(p * np.log(p))


H_human = (
    human_deduped.groupby("word.id")["probability_y"]
    .apply(shannon_entropy)
    .rename("H_human")
)

# ── Human unique counts and prob sums per context ────────────────────────
n_human_unique = (
    human_deduped.groupby("word.id")["answer_stripped"]
    .nunique()
    .rename("n_human_unique")
)
human_prob_sum = (
    human_deduped.groupby("word.id")["probability_y"]
    .sum()
    .rename("human_prob_sum")
)

# ── Model entropy per context (raw probs, not renormalized) ──────────────
# Dedup model predictions: groupby target_word_id + prediction_cleaned,
# keep highest probability_converted
gpt_sorted = gpt.sort_values("probability_converted", ascending=False)
gpt_model_deduped = gpt_sorted.drop_duplicates(
    subset=["target_word_id", "prediction_cleaned"], keep="first"
)

H_model = (
    gpt_model_deduped.groupby("target_word_id")["probability_converted"]
    .apply(shannon_entropy)
    .rename("H_model")
)

# ── Model unique counts and prob sums per context ────────────────────────
n_model_unique = (
    gpt_model_deduped.groupby("target_word_id")["prediction_cleaned"]
    .nunique()
    .rename("n_model_unique")
)
model_prob_sum = (
    gpt_model_deduped.groupby("target_word_id")["probability_converted"]
    .sum()
    .rename("model_prob_sum")
)

# ── Merge entropies ──────────────────────────────────────────────────────
entropy_df = pd.DataFrame({
    "H_human": H_human,
    "H_model": H_model,
    "n_human_unique": n_human_unique,
    "n_model_unique": n_model_unique,
    "human_prob_sum": human_prob_sum,
    "model_prob_sum": model_prob_sum,
}).reset_index()
entropy_df.columns = [
    "word_id", "H_human", "H_model",
    "n_human_unique", "n_model_unique",
    "human_prob_sum", "model_prob_sum",
]

# ── Save per_context_entropy.csv ─────────────────────────────────────────
per_context_entropy = (
    entropy_df.rename(columns={"word_id": "target_word"})
    .merge(context_lookup, on="target_word")
    [["left_context", "target_word", "H_human", "H_model",
      "n_human_unique", "n_model_unique", "human_prob_sum", "model_prob_sum"]]
)
per_context_entropy.to_csv(OUT / "per_context_entropy.csv", index=False)
print(f"Written {len(per_context_entropy)} rows to {OUT / 'per_context_entropy.csv'}")

# ── Quartile stratification on H_human ───────────────────────────────────
entropy_df["entropy_quartile"] = pd.qcut(
    entropy_df["H_human"], 4, labels=["Q1", "Q2", "Q3", "Q4"]
)

# ── Lemma-based overlap@K per context (Algorithm-0) ─────────────────────
# GPT: dedup by lemma, keep highest probability_converted per lemma per context
gpt_lemma_deduped = (
    gpt_sorted
    .drop_duplicates(subset=["target_word_id", "lemma_word"], keep="first")
    .groupby("target_word_id", sort=False)
    .apply(lambda df: df["lemma_word"].tolist(), include_groups=False)
    .rename("gpt_lemmas")
)

# Human: dedup surface answers, then group by lemma and sum probability
records = []
for word_id, gpt_lemmas in gpt_lemma_deduped.items():
    ppl = people[people["word.id"] == word_id]

    # Human lemma probabilities: surface dedup -> group by lemma -> sum
    answer_level = (
        ppl.assign(
            answer_stripped=ppl["answer"].str.strip(),
            lemma_stripped=ppl["lemma_answer"].astype(str).str.strip(),
        )
        .drop_duplicates(subset=["answer_stripped"])
        [["answer_stripped", "lemma_stripped", "probability_y"]]
    )
    lemma_prob = (
        answer_level.groupby("lemma_stripped")["probability_y"]
        .sum()
        .to_dict()
    )
    human_lemmas = set(lemma_prob.keys())

    row = {"word_id": word_id}

    for k in K_VALUES:
        top_k = set(gpt_lemmas[:k])
        found = human_lemmas & top_k
        row[f"overlap_at_{k}"] = (
            len(found) / len(human_lemmas) if human_lemmas else 0.0
        )
        row[f"weighted_overlap_at_{k}"] = sum(lemma_prob[l] for l in found)

    records.append(row)

overlap_df = pd.DataFrame(records)

# ── Merge everything ─────────────────────────────────────────────────────
result = entropy_df.merge(overlap_df, on="word_id")

# ── Save per_context_overlap.csv ─────────────────────────────────────────
overlap_cols = (
    ["left_context", "target_word", "entropy_quartile"]
    + [f"overlap_at_{k}" for k in K_VALUES]
    + [f"weighted_overlap_at_{k}" for k in K_VALUES]
)
per_context_overlap = (
    result.rename(columns={"word_id": "target_word"})
    .merge(context_lookup, on="target_word")
    [overlap_cols]
)
per_context_overlap.to_csv(OUT / "per_context_overlap.csv", index=False)
print(f"Written {len(per_context_overlap)} rows to {OUT / 'per_context_overlap.csv'}")

# ── Overall summary ──────────────────────────────────────────────────────
print(f"\nMean H_human = {result['H_human'].mean():.4f}")
print(f"Mean H_model = {result['H_model'].mean():.4f}\n")

# ── Quartile summary table ───────────────────────────────────────────────
print("=== Quartile Summary (Algorithm-0) ===\n")
cols = ["H_human", "H_model"] + [f"overlap_at_{k}" for k in K_VALUES] + [f"weighted_overlap_at_{k}" for k in K_VALUES]
summary = result.groupby("entropy_quartile", observed=True)[cols].mean()

# ── Save quartile_summary.csv ────────────────────────────────────────────
quartile_summary = summary.reset_index()
quartile_summary = quartile_summary.rename(columns={
    "entropy_quartile": "quartile",
    "H_human": "mean_H_human",
    "H_model": "mean_H_model",
})
quartile_summary.to_csv(OUT / "quartile_summary.csv", index=False)
print(f"Written {len(quartile_summary)} rows to {OUT / 'quartile_summary.csv'}\n")

header = f"{'Quartile':>8}  {'H_human':>8}  {'H_model':>8}"
for k in K_VALUES:
    header += f"  {'ov@'+str(k):>8}"
for k in K_VALUES:
    header += f"  {'wov@'+str(k):>8}"
print(header)
print("-" * len(header))

for q in ["Q1", "Q2", "Q3", "Q4"]:
    r = summary.loc[q]
    line = f"{q:>8}  {r['H_human']:>8.4f}  {r['H_model']:>8.4f}"
    for k in K_VALUES:
        line += f"  {r[f'overlap_at_{k}']:>8.3f}"
    for k in K_VALUES:
        line += f"  {r[f'weighted_overlap_at_{k}']:>8.3f}"
    print(line)
