== Target Word Analysis

Code: `analysis_target.py` · Data: `output/target/`

=== Method

To assess whether human and model predictions converge specifically on the _target_ word (the word originally deleted from the sentence), we computed the probability mass each source allocates to the target lemma. For humans, $p_"target"^"human"$ equals the summed response proportion (`probability_y`) across all unique answers whose lemma matches the target lemma within a given context. For the model, $p_"target"^"model"$ equals the summed converted probability (`probability_converted`) across all unique predictions whose lemma matches the target, after deduplication by surface form (retaining the highest-probability variant).

We correlated $p_"target"^"human"$ and $p_"target"^"model"$ across all 144 contexts using both Pearson and Spearman coefficients. To examine how target-word convergence relates to broader distributional overlap, we divided contexts into four classes based on whether each source produced the target lemma at all ($p_"target" > 0$):

- *C1 — Both hit:* both humans and the model produced the target.
- *C2 — Only humans:* at least one human respondent produced the target, but the model did not.
- *C3 — Only model:* the model produced the target, but no human respondent did.
- *C4 — Neither:* neither source produced the target.

For each class we report mean $p_"target"$, lemma-based overlap\@K, and probability-weighted overlap\@K.

=== Results

The correlation between human and model probability mass on the target lemma was very strong: Pearson $r = 0.88$ ($p < 10^(-47)$) and Spearman $r = 0.76$ ($p < 10^(-27)$). When humans strongly agree on the target word, the model also tends to assign it high probability, and vice versa.

@target_class_table shows the class breakdown. The largest class is C4 (86 of 144 contexts): in 60% of cases, neither humans nor the model produced the target word — these are contexts where the deleted word is not predictable from the left context alone. C1 contains 34 contexts where both sources hit the target; C2 has 22 where only humans did; C3 has just 2 where only the model did.

#figure(
  align(center)[
    #table(
      columns: (auto, auto, auto, auto, auto, auto),
      align: (center, center, center, center, center, center),
      table.header[Class][N][Mean $p_"target"^"human"$][Mean $p_"target"^"model"$][overlap\@10][w. overlap\@10],
      [C1 — both hit], [34], [0.384], [0.378], [0.335], [0.622],
      [C2 — only humans], [22], [0.096], [0.000], [0.146], [0.332],
      [C3 — only model], [2], [0.000], [0.092], [0.192], [0.581],
      [C4 — neither], [86], [0.000], [0.000], [0.193], [0.359],
    )
  ],
  caption: [Target-word convergence classes: class sizes, mean target-word probabilities, and overlap metrics.],
) <target_class_table>

To illustrate each class, consider the following examples (Russian contexts shown with the target word in parentheses):

- *C1 — both hit:* _"Причиной аварии был мобильный"_ (телефон). Humans assign $p = 0.95$, the model $p = 0.97$: a highly constraining context where both sources converge.
- *C2 — only humans:* _"Музыканты играли на похоронах, разгружали"_ (вагоны). Humans produce the target ($p = 0.32$) alongside _обстановку_ and _людей_ (each $p = 0.11$), but it does not appear among model predictions at all — the context requires pragmatic world knowledge that the model lacks.
- *C3 — only model:* _"У тебя впереди замечательный день,"_ (полный). No human respondent produced the target, yet the model ranks it second ($p = 0.12$) after the conjunction _и_ ($p = 0.22$), followed by the near-synonym _наполненный_ ($p = 0.07$). Only 2 contexts fall into this class.
- *C4 — neither:* _"В котёл бросают куски"_ (баранины). The target word is not recoverable from the left context alone; both humans and the model distribute their probability mass across other continuations.

=== Interpretation

The strong correlation ($r = 0.88$) between human and model target-word probability confirms that GPT-4o-mini captures the contextual predictability of target words with high fidelity.

The class-based analysis reveals a clear pattern. In C1, where both sources converge on the target, weighted overlap\@10 reaches 0.622 — indicating that shared success on the target word signals broader distributional alignment, not merely agreement on a single item. C2, where only humans hit the target (mean $p_"target"^"human" = 0.096$), shows the lowest overlap metrics (weighted overlap\@10 = 0.332). The human probability mass in these contexts is small, suggesting marginal or morphological-variant hits that the model misses.

C3 contains only 2 contexts, too few for reliable generalization, but notably has high weighted overlap (0.581) despite zero human target probability — the model recovers the target in contexts where humans diverge to other answers.

The dominant class is C4 (86 contexts, 60%): neither source guesses the target, yet weighted overlap\@10 is 0.359. This confirms that human–model distributional alignment does not depend on shared success at predicting the deleted word. Even when both fail on the target, they still converge on roughly a third of the probability mass over alternative continuations.
