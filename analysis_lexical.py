import pandas as pd
from pathlib import Path

K_VALUES = [5, 10, 15, 20, 50, 100, 200]
OUT = Path("output/lexical")
OUT.mkdir(parents=True, exist_ok=True)

people = pd.read_csv("people_with_prob.csv")
gpt = pd.read_csv("gpt4omini_morph_2.csv")

# --- Context lookup: target_word -> left_context ---
context_lookup = (
    people.drop_duplicates("word.id")[["word.id", "Left context"]]
    .rename(columns={"word.id": "target_word", "Left context": "left_context"})
)

# --- GPT: dedup by lemma, keep highest-prob per lemma per context ---
gpt_sorted = gpt.sort_values("probability_converted", ascending=False)
gpt_deduped = (
    gpt_sorted
    .drop_duplicates(subset=["target_word_id", "lemma_word"], keep="first")
    .groupby("target_word_id", sort=False)
    .apply(lambda df: df["lemma_word"].tolist(), include_groups=False)
    .rename("gpt_lemmas")
)

# --- GPT: dedup by surface form (for baseline comparison) ---
gpt["pred_stripped"] = gpt["prediction_cleaned"].str.strip()
gpt_sorted_surf = gpt.sort_values("probability_converted", ascending=False)
gpt_deduped_surf = (
    gpt_sorted_surf
    .drop_duplicates(subset=["target_word_id", "pred_stripped"], keep="first")
    .groupby("target_word_id", sort=False)
    .apply(lambda df: df["pred_stripped"].tolist(), include_groups=False)
    .rename("gpt_surface")
)

records = []
all_human_answers = []
all_human_lemmas = []

for word_id, gpt_lemmas in gpt_deduped.items():
    ppl = people[people["word.id"] == word_id]
    left_ctx = context_lookup.loc[
        context_lookup["target_word"] == word_id, "left_context"
    ].iloc[0]

    # --- Human lemmas: first get one probability per unique surface answer,
    #     then group by lemma and sum to get lemma-level probability ---
    answer_level = (
        ppl.assign(
            answer_stripped=ppl["answer"].str.strip(),
            lemma_stripped=ppl["lemma_answer"].astype(str).str.strip(),
        )
        .drop_duplicates(subset=["answer_stripped"])
        [["answer_stripped", "lemma_stripped", "probability_y"]]
    )

    # Collect human_answers rows
    for _, r in answer_level.iterrows():
        all_human_answers.append({
            "left_context": left_ctx,
            "target_word": word_id,
            "answer": r["answer_stripped"],
            "lemma_answer": r["lemma_stripped"],
            "probability_y": r["probability_y"],
        })

    lemma_prob = (
        answer_level.groupby("lemma_stripped")["probability_y"]
        .sum()
        .to_dict()
    )
    human_lemmas = set(lemma_prob.keys())

    # Collect human_lemmas rows
    for lemma, prob in lemma_prob.items():
        all_human_lemmas.append({
            "left_context": left_ctx,
            "target_word": word_id,
            "lemma": lemma,
            "probability": prob,
        })

    # --- Human surface answers (for baseline) ---
    answer_prob = (
        ppl.groupby(ppl["answer"].str.strip())["probability_y"]
        .first()
        .to_dict()
    )
    human_answers = set(answer_prob.keys())

    # --- GPT surface list for this context ---
    gpt_surf = gpt_deduped_surf.get(word_id, [])

    row = {
        "target_word": word_id,
        "n_human_lemmas": len(human_lemmas),
    }

    for k in K_VALUES:
        # Lemma-based overlap
        top_k_lemmas = set(gpt_lemmas[:k])
        found_lemmas = human_lemmas & top_k_lemmas
        row[f"overlap_at_{k}"] = (
            len(found_lemmas) / len(human_lemmas) if human_lemmas else 0.0
        )
        row[f"weighted_overlap_at_{k}"] = sum(
            lemma_prob[l] for l in found_lemmas
        )

        # Surface-form match@K (baseline)
        top_k_surf = set(gpt_surf[:k])
        found_surf = human_answers & top_k_surf
        row[f"surface_match_at_{k}"] = (
            len(found_surf) / len(human_answers) if human_answers else 0.0
        )
        row[f"surface_weighted_match_at_{k}"] = sum(
            answer_prob[a] for a in found_surf
        )

    records.append(row)

# --- Save human_answers.csv ---
ha_df = pd.DataFrame(all_human_answers)
ha_df.to_csv(OUT / "human_answers.csv", index=False)
print(f"Written {len(ha_df)} rows to {OUT / 'human_answers.csv'}")

# --- Save human_lemmas.csv ---
hl_df = pd.DataFrame(all_human_lemmas)
hl_df.to_csv(OUT / "human_lemmas.csv", index=False)
print(f"Written {len(hl_df)} rows to {OUT / 'human_lemmas.csv'}")

# --- Save per_context_overlap.csv ---
out = pd.DataFrame(records)
out = out.merge(context_lookup, left_on="target_word", right_on="target_word")
# Reorder so left_context and target_word are first
cols = ["left_context", "target_word"] + [
    c for c in out.columns if c not in ("left_context", "target_word")
]
out = out[cols]
out.to_csv(OUT / "per_context_overlap.csv", index=False)
print(f"Written {len(out)} rows to {OUT / 'per_context_overlap.csv'}")

# --- Save summary.csv ---
summary_records = []
for k in K_VALUES:
    summary_records.append({
        "K": k,
        "overlap_at_K": out[f"overlap_at_{k}"].mean(),
        "weighted_overlap_at_K": out[f"weighted_overlap_at_{k}"].mean(),
        "surface_match_at_K": out[f"surface_match_at_{k}"].mean(),
        "surface_weighted_match_at_K": out[f"surface_weighted_match_at_{k}"].mean(),
    })
summary = pd.DataFrame(summary_records)
summary.to_csv(OUT / "summary.csv", index=False)
print(f"Written {len(summary)} rows to {OUT / 'summary.csv'}")

# --- Print mean across all 144 contexts ---
print(f"\n=== Mean across all contexts ===\n")
print(f"{'K':>5}  {'overlap@K':>12}  {'wt_overlap@K':>14}  {'surf_match@K':>14}  {'wt_surf_match@K':>17}")
print("-" * 70)
for k in K_VALUES:
    ov = out[f"overlap_at_{k}"].mean()
    wov = out[f"weighted_overlap_at_{k}"].mean()
    sm = out[f"surface_match_at_{k}"].mean()
    wsm = out[f"surface_weighted_match_at_{k}"].mean()
    print(f"{k:>5}  {ov:>12.3f}  {wov:>14.3f}  {sm:>14.3f}  {wsm:>17.3f}")
