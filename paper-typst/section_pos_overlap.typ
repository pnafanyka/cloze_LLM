== POS Distribution Overlap

=== Method

To assess whether GPT-4o-mini captures the same syntactic preferences as human respondents, we compare the probability-weighted part-of-speech (POS) distributions produced by each source. For every sentence context, we construct two POS distributions:

- *Human POS distribution.* For each context, we first deduplicate human responses by surface form, retaining one row per unique answer with its cloze probability and Universal POS (UPOS) tag. We then sum the probabilities of all answers sharing the same POS tag, yielding a distribution over POS categories weighted by response frequency.

- *Model POS distribution.* Analogously, we deduplicate GPT-4o-mini predictions by surface form (keeping the highest-probability entry per word), then sum converted probabilities across predictions sharing the same UPOS tag.

We report two metrics. First, *POS overlap\@1*: for each context, we identify the POS tag with the highest total weight in each distribution and check whether the human and model top-POS tags match. The mean across all 144 contexts gives the overall POS overlap\@1 rate.

Second, we apply *Algorithm 1 (ranked POS intersection\@$k$)*. For each context, POS tags are ranked by descending total weight separately for human and model distributions. For $k = 1, 2, dots, 5$, we compute:

$ "intersection@" k = frac(|"top-" k "human POS" sect "top-" k "model POS"|, k) $

and average across all 144 contexts.

=== Results

The basic POS overlap\@1 rate is *0.556*, indicating that the model's most-weighted POS tag matches the human top POS in roughly 56% of contexts.

#align(center)[
#table(
  columns: (auto, auto),
  align: (center, center),
  table.header[$k$][mean POS intersection\@$k$],
  [1], [0.556],
  [2], [0.542],
  [3], [0.502],
  [4], [0.484],
  [5], [0.464],
)
]

Intersection\@$k$ decreases as $k$ grows, falling from 0.556 at $k = 1$ to 0.464 at $k = 5$. This decline indicates that beyond the dominant POS category, human and model distributions diverge in their secondary syntactic preferences.

=== Interpretation

A POS overlap\@1 of 56% shows that the model and humans agree on the dominant syntactic category roughly half the time. This is a moderate level of agreement --- substantially above chance (given the number of possible UPOS tags), yet far from ceiling. Contexts where they disagree often involve the model assigning highest mass to punctuation (PUNCT) or function words (ADP, SCONJ), while humans favour content-word categories such as PRON or VERB.

The gradual decline of intersection\@$k$ with increasing $k$ suggests that the syntactic diversity of human responses is only partially mirrored by the model. At $k = 5$ the model still recovers roughly 46% of the top human POS categories on average, indicating that the two distributions share a common core of syntactic expectations but diverge in the tails. These findings are consistent with the lexical overlap results: GPT-4o-mini captures the broad syntactic shape of human expectations while differing in finer distributional details.
