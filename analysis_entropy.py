import numpy as np
import pandas as pd
from pathlib import Path

K_VALUES = [5, 10, 20, 50, 100]

people = pd.read_csv("people_with_prob.csv")
gpt = pd.read_csv("gpt4omini_morph_2.csv")

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

# ── Merge entropies ──────────────────────────────────────────────────────
entropy_df = pd.DataFrame({"H_human": H_human, "H_model": H_model}).reset_index()
entropy_df.columns = ["word_id", "H_human", "H_model"]

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

Path("output").mkdir(exist_ok=True)
result.to_csv("output/entropy_analysis.csv", index=False)
print(f"Written {len(result)} rows to output/entropy_analysis.csv\n")

# ── Overall summary ──────────────────────────────────────────────────────
print(f"Mean H_human = {result['H_human'].mean():.4f}")
print(f"Mean H_model = {result['H_model'].mean():.4f}\n")

# ── Quartile summary table ───────────────────────────────────────────────
print("=== Quartile Summary (Algorithm-0) ===\n")
cols = ["H_human", "H_model"] + [f"overlap_at_{k}" for k in K_VALUES] + [f"weighted_overlap_at_{k}" for k in K_VALUES]
summary = result.groupby("entropy_quartile", observed=True)[cols].mean()

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
