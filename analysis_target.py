from pathlib import Path

import pandas as pd
from scipy import stats

from filter_data import load_gpt, load_human

K_VALUES = [5, 10, 20, 50, 100]
OUT = Path("output/target")
OUT.mkdir(parents=True, exist_ok=True)

people = load_human()
gpt = load_gpt()

# ---- Context lookup table ----
context_lookup = (
    people.drop_duplicates("word.id")[["word.id", "Left context"]]
    .rename(columns={"word.id": "target_word", "Left context": "left_context"})
)

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

# n_human_target_matches: count of unique human answers where lemma_accuracy == 1
n_human_target = (
    human_deduped[human_deduped["lemma_accuracy"] == 1]
    .groupby("word.id")["answer_stripped"]
    .nunique()
    .reindex(people["word.id"].unique(), fill_value=0)
    .rename("n_human_target_matches")
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

# n_model_target_matches: count of unique model predictions where lemma_accuracy == 1
n_model_target = (
    gpt_deduped[gpt_deduped["lemma_accuracy"] == 1]
    .groupby("target_word_id")["pred_stripped"]
    .nunique()
    .reindex(people["word.id"].unique(), fill_value=0)
    .rename("n_model_target_matches")
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
contexts = contexts.merge(
    n_human_target.reset_index().rename(columns={"word.id": "word_id"}),
    on="word_id",
    how="left",
)
contexts = contexts.merge(
    n_model_target.reset_index().rename(columns={"target_word_id": "word_id"}),
    on="word_id",
    how="left",
)
contexts["p_target_human"] = contexts["p_target_human"].fillna(0.0)
contexts["p_target_model"] = contexts["p_target_model"].fillna(0.0)
contexts["n_human_target_matches"] = contexts["n_human_target_matches"].fillna(0).astype(int)
contexts["n_model_target_matches"] = contexts["n_model_target_matches"].fillna(0).astype(int)

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

# ---- Save correlation_summary.csv ----
corr_df = pd.DataFrame([
    {"metric": "Pearson", "r": r_pearson, "p_value": p_pearson},
    {"metric": "Spearman", "r": r_spearman, "p_value": p_spearman},
])
corr_df.to_csv(OUT / "correlation_summary.csv", index=False)

# ---- 2×2 target-hit classes ----
human_hit = contexts["p_target_human"] > 0
model_hit = contexts["p_target_model"] > 0
contexts["target_class"] = "C4"  # default: neither
contexts.loc[human_hit & model_hit, "target_class"] = "C1"   # both hit
contexts.loc[human_hit & ~model_hit, "target_class"] = "C2"  # only humans
contexts.loc[~human_hit & model_hit, "target_class"] = "C3"  # only model
contexts.loc[~human_hit & ~model_hit, "target_class"] = "C4"  # neither

# ---- Save per_context_target_probs.csv ----
probs_out = contexts[["word_id", "p_target_human", "p_target_model",
                       "n_human_target_matches", "n_model_target_matches"]].copy()
probs_out = probs_out.rename(columns={"word_id": "target_word"})
probs_out.insert(1, "target_lemma", probs_out["target_word"])
probs_out = probs_out.merge(context_lookup, on="target_word", how="left")
# Reorder: left_context, target_word first
probs_out = probs_out[["left_context", "target_word", "target_lemma",
                        "p_target_human", "p_target_model",
                        "n_human_target_matches", "n_model_target_matches"]]
probs_out.to_csv(OUT / "per_context_target_probs.csv", index=False)

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
        rec[f"weighted_overlap_at_{k}"] = sum(h_probs[lem] for lem in found)

    overlap_records.append(rec)

overlap_df = pd.DataFrame(overlap_records)
contexts = pd.concat([contexts.reset_index(drop=True), overlap_df], axis=1)

# ---- Save per_context_overlap.csv ----
overlap_cols = (
    ["word_id", "target_class"]
    + [f"overlap_at_{k}" for k in K_VALUES]
    + [f"weighted_overlap_at_{k}" for k in K_VALUES]
)
overlap_out = contexts[overlap_cols].copy()
overlap_out = overlap_out.rename(columns={"word_id": "target_word"})
overlap_out = overlap_out.merge(context_lookup, on="target_word", how="left")
overlap_out = overlap_out[
    ["left_context", "target_word", "target_class"]
    + [f"overlap_at_{k}" for k in K_VALUES]
    + [f"weighted_overlap_at_{k}" for k in K_VALUES]
]
overlap_out.to_csv(OUT / "per_context_overlap.csv", index=False)

# Also remove stale quartile file if present
(OUT / "quartile_summary.csv").unlink(missing_ok=True)

print(f"Written {len(contexts)} rows to {OUT}/\n")

# ---- Class summary ----
CLASS_LABELS = ["C1", "C2", "C3", "C4"]
CLASS_DESCR = {
    "C1": "both hit",
    "C2": "only humans",
    "C3": "only model",
    "C4": "neither",
}

print("=== Class counts ===")
for c in CLASS_LABELS:
    n = (contexts["target_class"] == c).sum()
    print(f"  {c} ({CLASS_DESCR[c]}): {n}")
print()

print("=== Class summary (target-hit classes) ===\n")
summary_cols = ["p_target_human", "p_target_model"]
summary_cols += [f"overlap_at_{k}" for k in K_VALUES]
summary_cols += [f"weighted_overlap_at_{k}" for k in K_VALUES]
q_summary = contexts.groupby("target_class", observed=True)[summary_cols].mean()

# ---- Save class_summary.csv ----
q_out = q_summary.reset_index().rename(columns={
    "target_class": "class",
    "p_target_human": "mean_p_target_human",
    "p_target_model": "mean_p_target_model",
})
q_out["description"] = q_out["class"].map(CLASS_DESCR)
q_out["n_contexts"] = [
    (contexts["target_class"] == c).sum() for c in q_out["class"]
]
q_out.to_csv(OUT / "class_summary.csv", index=False)

print(f"{'Class':>10}  {'mean_p_h':>10}  {'mean_p_m':>10}", end="")
for k in K_VALUES:
    print(f"  {'ov@' + str(k):>8}", end="")
for k in K_VALUES:
    print(f"  {'w_ov@' + str(k):>10}", end="")
print()
print("-" * 130)
for c in CLASS_LABELS:
    if c not in q_summary.index:
        continue
    r = q_summary.loc[c]
    print(f"{c:>10}  {r['p_target_human']:>10.4f}  {r['p_target_model']:>10.4f}", end="")
    for k in K_VALUES:
        print(f"  {r[f'overlap_at_{k}']:>8.3f}", end="")
    for k in K_VALUES:
        print(f"  {r[f'weighted_overlap_at_{k}']:>10.3f}", end="")
    print()
