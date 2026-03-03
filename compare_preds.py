import pandas as pd
from pathlib import Path

K_VALUES = [1, 5, 10, 20, 50, 100, 200, 500, 1000]

people = pd.read_csv("people_with_prob.csv")
gpt = pd.read_csv("gpt4omini_morph_2.csv")

# Strip whitespace from prediction tokens
gpt["pred_stripped"] = gpt["prediction_cleaned"].str.strip()

# Sort GPT rows by probability descending, then deduplicate per context
# keeping the highest-probability occurrence of each stripped prediction
gpt_sorted = gpt.sort_values("probability_converted", ascending=False)
gpt_deduped = (
    gpt_sorted
    .drop_duplicates(subset=["target_word_id", "pred_stripped"], keep="first")
    .groupby("target_word_id", sort=False)
    .apply(lambda df: df["pred_stripped"].tolist(), include_groups=False)
    .rename("gpt_ranked")
)

records = []
for word_id, gpt_preds in gpt_deduped.items():
    ppl = people[people["word.id"] == word_id]

    # Unique human answers and their probabilities
    # probability_y is the same for all rows with the same answer, so take first
    answer_prob = (
        ppl.groupby(ppl["answer"].str.strip())["probability_y"]
        .first()
        .to_dict()
    )
    human_answers = set(answer_prob.keys())
    n_respondents = len(ppl)

    # Human accuracy on ground truth
    correct = ppl[ppl["cloze_accuracy"] == 1]
    human_accuracy = float(correct["probability_y"].iloc[0]) if len(correct) > 0 else 0.0

    left_context = ppl["Left context"].iloc[0]

    row = {
        "word_id": word_id,
        "left_context": left_context,
        "n_human_respondents": n_respondents,
        "n_human_unique_answers": len(human_answers),
        "human_accuracy": human_accuracy,
    }

    # Build match@k columns
    gpt_pred_list = gpt_preds  # already deduplicated ordered list
    for k in K_VALUES:
        top_k = set(gpt_pred_list[:k])
        found = human_answers & top_k
        row[f"match_at_{k}"] = len(found) / len(human_answers) if human_answers else 0.0
        row[f"weighted_match_at_{k}"] = sum(answer_prob[a] for a in found)

    records.append(row)

out = pd.DataFrame(records)
Path("output").mkdir(exist_ok=True)
out.to_csv("output/comparison.csv", index=False)
print(f"Written {len(out)} rows to output/comparison.csv")
print(out[["word_id", "human_accuracy", "match_at_10", "match_at_100", "weighted_match_at_100"]].to_string())
