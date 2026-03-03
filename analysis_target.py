import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

K_VALUES = [5, 10, 20, 50, 100]

people = pd.read_csv("people_with_prob.csv")
gpt = pd.read_csv("gpt4omini_morph_2.csv")

# ---- p_target_human per context ----
# Dedup human answers: groupby word.id + answer, take first probability_y and lemma_accuracy
people["answer_stripped"] = people["answer"].str.strip()
human_deduped = people.drop_duplicates(subset=["word.id", "answer_stripped"], keep="first")
# p_target_human = sum of probability_y where lemma_accuracy == 1
p_target_human = (
    human_deduped[human_deduped["lemma_accuracy"] == 1]
    .groupby("word.id")["probability_y"]
    .sum()
    .reindex(people["word.id"].unique(), fill_value=0.0)
    .rename("p_target_human")
)

# ---- p_target_model per context ----
# Dedup model predictions: groupby target_word_id + prediction_cleaned, keep highest probability_converted
gpt["pred_stripped"] = gpt["prediction_cleaned"].str.strip()
gpt_sorted = gpt.sort_values("probability_converted", ascending=False)
gpt_deduped = gpt_sorted.drop_duplicates(
    subset=["target_word_id", "pred_stripped"], keep="first"
)
# p_target_model = sum of probability_converted where lemma_accuracy == 1
p_target_model = (
    gpt_deduped[gpt_deduped["lemma_accuracy"] == 1]
    .groupby("target_word_id")["probability_converted"]
    .sum()
    .reindex(people["word.id"].unique(), fill_value=0.0)
    .rename("p_target_model")
)

# ---- Merge into a per-context frame ----
contexts = pd.DataFrame({"word_id": people["word.id"].unique()})
contexts = contexts.merge(
    p_target_human.reset_index().rename(columns={"word.id": "word_id"}),
    on="word_id",
    how="left",
)
contexts = contexts.merge(
    p_target_model.reset_index().rename(columns={"target_word_id": "word_id"}),
    on="word_id",
    how="left",
)
contexts["p_target_human"] = contexts["p_target_human"].fillna(0.0)
contexts["p_target_model"] = contexts["p_target_model"].fillna(0.0)

# ---- Correlation ----
r_pearson, p_pearson = stats.pearsonr(
    contexts["p_target_human"], contexts["p_target_model"]
)
r_spearman, p_spearman = stats.spearmanr(
    contexts["p_target_human"], contexts["p_target_model"]
)

print("=== Correlation: p_target_human vs p_target_model ===")
print(f"  Pearson  r = {r_pearson:.4f}, p = {p_pearson:.2e}")
print(f"  Spearman r = {r_spearman:.4f}, p = {p_spearman:.2e}")
print()

# ---- Quartiles by p_target_human ----
# Many contexts have p_target_human == 0, so we use rank-based quartiles
# with duplicates="drop" to handle tied zero values.
contexts["target_quartile"] = pd.qcut(
    contexts["p_target_human"].rank(method="first"),
    4,
    labels=["Q1", "Q2", "Q3", "Q4"],
)

# ---- Overlap@K computation per context ----
# Prepare GPT lemmas: dedup by lemma, keep highest prob per lemma per context
gpt_lemma_deduped = (
    gpt_sorted.drop_duplicates(subset=["target_word_id", "lemma_word"], keep="first")
    .groupby("target_word_id", sort=False)
    .apply(lambda df: df["lemma_word"].tolist(), include_groups=False)
    .rename("gpt_lemmas")
)

# Prepare human lemma probabilities per context
# First get one probability per unique surface answer, then aggregate by lemma
human_lemma_probs = {}
human_lemma_sets = {}
for word_id, grp in people.groupby("word.id"):
    answer_level = (
        grp.assign(
            answer_stripped=grp["answer"].str.strip(),
            lemma_stripped=grp["lemma_answer"].astype(str).str.strip(),
        )
        .drop_duplicates(subset=["answer_stripped"])
        [["lemma_stripped", "probability_y"]]
    )
    lemma_prob = answer_level.groupby("lemma_stripped")["probability_y"].sum().to_dict()
    human_lemma_probs[word_id] = lemma_prob
    human_lemma_sets[word_id] = set(lemma_prob.keys())

# Compute overlap metrics per context
overlap_records = []
for _, row in contexts.iterrows():
    wid = row["word_id"]
    rec = {}
    gpt_lemmas = gpt_lemma_deduped.get(wid, [])
    h_lemmas = human_lemma_sets.get(wid, set())
    h_probs = human_lemma_probs.get(wid, {})

    for k in K_VALUES:
        top_k = set(gpt_lemmas[:k])
        found = h_lemmas & top_k
        rec[f"overlap_at_{k}"] = len(found) / len(h_lemmas) if h_lemmas else 0.0
        rec[f"weighted_overlap_at_{k}"] = sum(h_probs[l] for l in found)

    overlap_records.append(rec)

overlap_df = pd.DataFrame(overlap_records)
contexts = pd.concat([contexts.reset_index(drop=True), overlap_df], axis=1)

# ---- Save output ----
Path("output").mkdir(exist_ok=True)
out_cols = (
    ["word_id", "p_target_human", "p_target_model", "target_quartile"]
    + [f"overlap_at_{k}" for k in K_VALUES]
    + [f"weighted_overlap_at_{k}" for k in K_VALUES]
)
contexts[out_cols].to_csv("output/target_analysis.csv", index=False)
print(f"Written {len(contexts)} rows to output/target_analysis.csv\n")

# ---- Quartile summary ----
print("=== Quartile summary (by p_target_human) ===\n")
summary_cols = ["p_target_human", "p_target_model"]
summary_cols += [f"overlap_at_{k}" for k in K_VALUES]
summary_cols += [f"weighted_overlap_at_{k}" for k in K_VALUES]
q_summary = contexts.groupby("target_quartile", observed=True)[summary_cols].mean()

print(f"{'Quartile':>10}  {'mean_p_h':>10}  {'mean_p_m':>10}", end="")
for k in K_VALUES:
    print(f"  {'ov@' + str(k):>8}", end="")
for k in K_VALUES:
    print(f"  {'w_ov@' + str(k):>10}", end="")
print()
print("-" * 130)
for q in ["Q1", "Q2", "Q3", "Q4"]:
    r = q_summary.loc[q]
    print(f"{q:>10}  {r['p_target_human']:>10.4f}  {r['p_target_model']:>10.4f}", end="")
    for k in K_VALUES:
        print(f"  {r[f'overlap_at_{k}']:>8.3f}", end="")
    for k in K_VALUES:
        print(f"  {r[f'weighted_overlap_at_{k}']:>10.3f}", end="")
    print()
