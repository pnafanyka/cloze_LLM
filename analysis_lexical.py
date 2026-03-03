import pandas as pd
from pathlib import Path

K_VALUES = [5, 10, 15, 20, 50, 100, 200]

people = pd.read_csv("people_with_prob.csv")
gpt = pd.read_csv("gpt4omini_morph_2.csv")

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
for word_id, gpt_lemmas in gpt_deduped.items():
    ppl = people[people["word.id"] == word_id]

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
    lemma_prob = (
        answer_level.groupby("lemma_stripped")["probability_y"]
        .sum()
        .to_dict()
    )
    human_lemmas = set(lemma_prob.keys())

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
        "word_id": word_id,
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

out = pd.DataFrame(records)
Path("output").mkdir(exist_ok=True)
out.to_csv("output/lexical_overlap.csv", index=False)

print(f"Written {len(out)} rows to output/lexical_overlap.csv\n")

# --- Print mean across all 144 contexts ---
print("=== Mean across all contexts ===\n")
print(f"{'K':>5}  {'overlap@K':>12}  {'wt_overlap@K':>14}  {'surf_match@K':>14}  {'wt_surf_match@K':>17}")
print("-" * 70)
for k in K_VALUES:
    ov = out[f"overlap_at_{k}"].mean()
    wov = out[f"weighted_overlap_at_{k}"].mean()
    sm = out[f"surface_match_at_{k}"].mean()
    wsm = out[f"surface_weighted_match_at_{k}"].mean()
    print(f"{k:>5}  {ov:>12.3f}  {wov:>14.3f}  {sm:>14.3f}  {wsm:>17.3f}")
