== Lexical Overlap (Lemma-Based)

Code: `analysis_lexical.py` · Data: `output/lexical_overlap.csv`

=== Method

The surface-form match\@$k$ metric reported earlier treats morphological variants of the same
word as distinct types. In a morphologically rich language such as Russian, this penalises
both humans and the model for producing different inflected forms of the same lexeme. To
address this, we compute a lemma-based overlap metric.

GPT-4o-mini predictions were deduplicated by lemma: for each context, predictions were sorted
by descending probability and only the highest-probability occurrence of each unique lemma was
retained. Human responses were similarly aggregated at the lemma level: for each context,
unique surface-form answers were first identified with their cloze probabilities, then grouped
by lemma, summing probabilities across surface forms that share the same lemma.

Two metrics were computed for $k in {5, 10, 15, 20, 50, 100, 200}$:

- *overlap\@$k$* $=$ $|L_"human" sect L^k_"GPT"| slash |L_"human"|$, where $L_"human"$ is
  the set of unique human lemmas and $L^k_"GPT"$ is the set of top-$k$ GPT lemmas.
- *weighted overlap\@$k$* $=$ $sum_(l in L_"human" sect L^k_"GPT") p(l)$, where $p(l)$ is
  the summed cloze probability of lemma $l$.

=== Results

#align(center)[
#table(
  columns: (auto, auto, auto),
  align: (center, center, center),
  table.header[$k$][overlap\@$k$][weighted overlap\@$k$],
  [5],   [0.188], [0.386],
  [10],  [0.219], [0.420],
  [15],  [0.225], [0.425],
  [20],  [0.227], [0.429],
  [50],  [0.232], [0.434],
  [100], [0.233], [0.435],
  [200], [0.235], [0.436],
)
]

For comparison, the corresponding surface-form metrics are reproduced below:

#align(center)[
#table(
  columns: (auto, auto, auto),
  align: (center, center, center),
  table.header[$k$][surface match\@$k$][weighted surface match\@$k$],
  [5],   [0.176], [0.363],
  [10],  [0.208], [0.398],
  [20],  [0.213], [0.404],
  [100], [0.216], [0.406],
)
]

=== Interpretation

Lemma-based overlap consistently exceeds surface-form matching at every $k$, confirming that
a portion of apparent mismatches between human and model responses are attributable to
morphological variation rather than genuine lexical divergence. At $k = 10$, lemma overlap
reaches 0.219 (vs.\ 0.208 for surface match) and weighted lemma overlap reaches 0.420
(vs.\ 0.398), representing a relative improvement of roughly 5% in type coverage and
6% in probability mass coverage.

However, the gains from lemmatisation are modest. The lemma-based ceiling at $k = 200$
(0.235 unweighted, 0.436 weighted) remains well below full coverage, indicating that the
dominant source of human–model divergence is not inflectional variation but rather genuine
lexical disagreement: humans and the model frequently propose different words altogether.

The saturation pattern mirrors that of the surface-form metric: nearly all improvement
occurs before $k = 20$, with negligible gains beyond that point. This reinforces the
finding that GPT-4o-mini's useful vocabulary of continuations is effectively exhausted
within its top-20 predictions.
