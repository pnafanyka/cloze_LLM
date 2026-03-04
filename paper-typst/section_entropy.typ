== Entropy-Stratified Analysis

Code: `analysis_entropy.py` · Data: `output/entropy/`

=== Method

To assess how distributional uncertainty in human responses relates to human--model agreement, we compute Shannon entropy for each of the 144 sentence contexts and stratify the overlap analysis by entropy quartile.

*Human entropy.* For each context $c$, let $p_1, dots, p_m$ be the cloze probabilities of the $m$ distinct answers given by respondents (these proportions sum to 1 within each context). The Shannon entropy is

$ H_"human" (c) = - sum_(i=1)^(m) p_i ln p_i $

Low entropy indicates high agreement among respondents (one dominant answer); high entropy indicates a dispersed response distribution.

*Model entropy.* For GPT-4o-mini, the API returns an unnormalized probability distribution over multi-token continuations: the sum of `probability_converted` per context ranges from 0.13 to 2.00 (median 0.51), with 7 of 144 contexts exceeding 1.0.#footnote[Sums exceeding 1.0 arise because the model predictions include both partial prefixes and their completions as separate entries. For example, for the context targeting _банке_, both the prefix "бан" ($p = 0.996$) and the full continuation "банке" ($p = 0.996$) appear as distinct rows --- their probabilities are per-path in the token tree, not per-leaf, so they are not mutually exclusive. The same pattern occurs in all 7 affected contexts (e.g., "покрыв"/"покрывало", "раст"/"растительным").] We compute entropy over the raw (non-renormalized) probabilities:

$ H_"model" (c) = - sum_(j=1)^(n) q_j ln q_j $

where $q_j$ is `probability_converted` for the $j$-th unique prediction. Because the distribution is neither complete nor guaranteed to sum to 1, $H_"model"$ is not directly comparable in magnitude to $H_"human"$, but it still captures how concentrated or spread the model's probability mass is within the observed portion.

*Quartile stratification.* We rank all 144 contexts by $H_"human"$ and split them into four quartiles using `pandas.qcut` (Q1 = lowest entropy, Q4 = highest). Within each quartile we report the mean lemma-based overlap\@$K$ and weighted overlap\@$K$ at $K in {5, 10, 20, 50, 100}$, following the same lemma-matching procedure described in the lexical overlap analysis.

=== Results

Across all 144 contexts, the mean human entropy is $H_"human" = 2.25$ nats and the mean model entropy is $H_"model" = 1.00$ nats. The model's partial distribution is substantially more concentrated than the human response distribution.

#align(center)[
#table(
  columns: (auto, auto, auto, auto, auto),
  align: (center, center, center, center, center),
  table.header[Quartile][Mean $H_"human"$][Mean $H_"model"$][overlap\@10][weighted overlap\@10],
  [Q1 (low)], [1.15], [0.93], [0.379], [0.747],
  [Q2], [2.05], [1.11], [0.256], [0.451],
  [Q3], [2.55], [1.05], [0.145], [0.261],
  [Q4 (high)], [3.25], [0.90], [0.096], [0.222],
)
]

The pattern is monotonic: as human uncertainty increases from Q1 to Q4, lemma overlap\@10 drops from 37.9% to 9.6%, and weighted overlap\@10 drops from 74.7% to 22.2%. A similar trend holds across all values of $K$: even at $K = 100$, Q4 contexts reach only 12.0% overlap compared to 39.2% for Q1.

Notably, model entropy does not increase in parallel with human entropy. Q4 contexts (highest human uncertainty) actually have the _lowest_ mean model entropy (0.90 nats), while Q2 contexts show the highest (1.11 nats). The model remains confident regardless of whether human respondents agree or disagree.

=== Interpretation

The entropy-stratified analysis reveals a clear asymmetry between human and model uncertainty. When humans converge on a small set of likely completions (low-entropy contexts), the model's top predictions overlap substantially with the human response set. When human responses are highly dispersed, the model fails to capture the long tail of plausible completions, and overlap drops sharply.

This dissociation is not simply a floor effect. Weighted overlap, which accounts for the probability mass of matched answers, shows the same gradient: the model captures 74.7% of the human probability mass in Q1 contexts but only 22.2% in Q4. This means the model is not merely missing rare answers in high-entropy contexts --- it is also failing to predict the more probable responses.

The finding that model entropy is essentially flat (or even slightly _lower_ in high-entropy contexts) suggests that GPT-4o-mini does not modulate its confidence in response to genuine predictive difficulty. Where humans hedge by distributing probability across many completions, the model maintains a concentrated distribution, producing a narrow beam of high-confidence predictions. This pattern is consistent with the known tendency of autoregressive language models to be overconfident and underestimate the entropy of natural language.
