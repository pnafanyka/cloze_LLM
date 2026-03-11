== Surface-Form Overlap

Code: `compare_preds.py` · Data: `output/comparison.csv`

=== Method

GPT-4o-mini predictions were sorted by descending probability and deduplicated
by stripped surface form, retaining the highest-probability occurrence of each
unique token per context. Human responses were likewise deduplicated by stripped
surface form, with each unique answer retaining its cloze probability.

Two metrics were computed for $k in {1, 5, 10, 20, 100, 1000}$:

- *match\@$k$* — the fraction of *unique* human answer types that appear
  anywhere in GPT's top-$k$ predictions (unweighted).
- *weighted match\@$k$* — the fraction of *human response mass* covered: for each human answer
  found in the top-$k$, its probability (response frequency) is summed.
  Because popular answers contribute more, weighted match is consistently higher than
  unweighted match and is the more meaningful metric.

=== Results

Coverage saturates rapidly: almost all gains occur before $k = 10$, with negligible
improvement beyond $k = 20$.

#align(center)[
#table(
  columns: (auto, auto, auto),
  align: (center, center, center),
  table.header[$k$][match\@$k$][weighted match\@$k$],
  [1],   [0.064], [0.180],
  [5],   [0.176], [0.363],
  [10],  [0.208], [0.398],
  [20],  [0.213], [0.404],
  [100], [0.216], [0.406],
  [1000],[0.216], [0.406],
)
]

At $k = 10$ the model's top candidates already cover 40 % of human response mass on average,
but only 21 % of unique answer *types* — reflecting that GPT reliably finds the most frequent
human answers while missing rare or idiosyncratic ones.

In 9 contexts (e.g. _Когда она в самолёте_, _В резервациях_, _У Пашки_) GPT produces *zero overlap* with
any human answer even at $k = 1000$, suggesting fundamental distributional divergence on
semantically constrained or low-frequency items.

=== Interpretation

GPT-4o-mini's probability mass concentrates on the same words humans prefer most, but its
vocabulary of plausible continuations is much narrower: the long tail of human creativity
(typos aside) is largely absent from the model's distribution.
The sharp saturation of match\@k by $k = 10$ means that simply taking the model's
top-10 candidates is nearly as informative as its entire ranked list.
