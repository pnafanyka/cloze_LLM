# Data Files

## people_with_prob.csv — Human cloze norms
6,898 rows across 144 Russian sentence contexts (`word.id`).
Each row = one participant's answer to one context.
628 unique subjects; 14–151 per context (avg ~48); 1–65 unique answers per context.
`probability_y` = proportion of respondents who gave that answer for that context (pre-computed, repeated for rows sharing the same answer).
Includes morphological annotation of each answer (Stanza): `upos_answer`, `feats_answer`, `mapped_feats_ud`, etc.
Accuracy columns (`cloze_accuracy`, `lemma_accuracy`, `pos_accuracy`, `feature_accuracy`) compare each answer against the ground-truth target word.

## gpt4omini_morph_2.csv — GPT-4o-mini predictions (main model file)
214,092 rows across the same 144 contexts (`target_word_id` = `word.id`).
Each row = one ranked token continuation from GPT-4o-mini, sorted by descending probability.
`sum_logprobs` is the raw log-prob; `probability_converted = exp(sum_logprobs)`.
6–13,957 rows per context (avg ~1,487); probabilities do NOT sum to 1 (range 0.13–2.00, median 0.51; 7 contexts exceed 1.0 due to overlapping multi-token paths).
Post-hoc Stanza morphological tagging added as `upos_word`, `lemma_word`, `feats`.
Top predictions are often punctuation or pronouns, not the target word.

Both CSVs have an `is_russian` column. All analysis scripts use `filter_data.py` to load data filtered to `is_russian == True`, yielding 6,705 human rows (622 subjects, 11–147 per context) and 21,818 GPT rows (2–1,067 candidates per context after dedup). Human `probability_y` is renormalized after filtering.

## combined_results_gpt4o_mini.csv — Raw GPT-4o-mini output (pre-processing)
Simpler, earlier-stage file with only 6 columns: `row_number`, `text_test`, `finished`, `prediction_cleaned`, `sum_logprobs`, `probability_converted`.
No `target_word_id` or morphological info — this is the raw API dump before the morphological markup and accuracy scoring pipeline was applied.
Covers the same 144 contexts as the full file.

## llama_with_all.csv — Llama predictions
Similar structure to `gpt4omini_morph_2.csv` but from a Llama-based model.
Context text uses `<mask>` placeholder format instead of a plain truncated sentence.
Additional columns: `probability`, `token_probabilities`, `token_count`, `depth`, `separator_probability`.
Same 144 contexts; also includes Stanza morphological annotation and accuracy columns.
