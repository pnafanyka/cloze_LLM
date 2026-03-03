== Target Word Analysis

Code: `analysis_target.py` · Data: `output/target_analysis.csv`

=== Method

To assess whether human and model predictions converge specifically on the _target_ word (the word originally deleted from the sentence), we computed the probability mass each source allocates to the target lemma. For humans, $p_"target"^"human"$ equals the summed response proportion (`probability_y`) across all unique answers whose lemma matches the target lemma within a given context. For the model, $p_"target"^"model"$ equals the summed converted probability (`probability_converted`) across all unique predictions whose lemma matches the target, after deduplication by surface form (retaining the highest-probability variant).

We correlated $p_"target"^"human"$ and $p_"target"^"model"$ across all 144 contexts using both Pearson and Spearman coefficients. To examine how target-word convergence relates to broader distributional overlap, we divided contexts into quartiles by $p_"target"^"human"$ (using rank-based binning, since 88 of 144 contexts have $p_"target"^"human" = 0$) and computed mean lemma-based overlap\@K and probability-weighted overlap\@K within each quartile.

=== Results

The correlation between human and model probability mass on the target lemma was very strong: Pearson $r = 0.88$ ($p < 10^(-47)$) and Spearman $r = 0.76$ ($p < 10^(-27)$). When humans strongly agree on the target word, the model also tends to assign it high probability, and vice versa.

@target_quartile_table shows the quartile breakdown. Contexts in Q4, where humans allocate on average 40% of their response mass to the target lemma, show the highest overlap with model predictions at every K value. Weighted overlap\@10 in Q4 (0.580) is substantially higher than in Q1--Q3 (0.340--0.399), indicating that contexts with strong human convergence on the target also enjoy broader alignment between human and model distributions.

#figure(
  align(center)[
    #table(
      columns: (auto, auto, auto, auto, auto),
      align: (center, center, center, center, center),
      table.header[Quartile][Mean $p_"target"^"human"$][Mean $p_"target"^"model"$][overlap\@10][weighted overlap\@10],
      [Q1], [0.000], [0.000], [0.172], [0.363],
      [Q2], [0.000], [0.005], [0.188], [0.340],
      [Q3], [0.017], [0.009], [0.223], [0.399],
      [Q4], [0.404], [0.348], [0.293], [0.580],
    )
  ],
  caption: [Mean target-word probability and overlap metrics by quartile of human target convergence ($p_"target"^"human"$).],
) <target_quartile_table>

=== Interpretation

The strong correlation ($r = 0.88$) between human and model target-word probability demonstrates that GPT-4o-mini captures the contextual predictability of target words with high fidelity: contexts that make the target word highly predictable for humans are also contexts where the model assigns it high probability.

The quartile analysis reveals an important asymmetry. In the majority of contexts (Q1--Q2, 72 of 144), neither humans nor the model converge on the target lemma --- these are contexts where the deleted word is not easily predictable from the left context alone. Yet even in these low-predictability contexts, lemma overlap\@10 remains around 17--19%, suggesting that model--human alignment is not driven solely by shared success at guessing the target.

In Q4, where the target is highly predictable, overlap jumps to 29% and weighted overlap to 58%. This shows that human convergence on the target word acts as a strong signal of broader distributional agreement: when humans agree on a specific continuation, the model's probability distribution aligns more closely with the full human response distribution, not just with the target itself.
