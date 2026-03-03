#align(center)[
  #text(17pt)[*Human vs GPT-4o-mini Cloze Predictions*]

  #v(6pt)
  #datetime.today().display()
]

#v(10pt)

= Task and Data

A *cloze task* presents a sentence with the final word removed; respondents supply what comes next.
The dataset covers *144 Russian sentence contexts*.
Human responses were collected from 628 subjects (14–151 per context, avg 48),
yielding 6 898 individual responses with 1–65 unique answers per context.
GPT-4o-mini predictions were obtained as a ranked list of token continuations
with log-probabilities, converted to probabilities via $exp(sum "logprob")$.
After deduplication by stripped surface form, each context has between 6 and 13 957 distinct GPT candidates.

= Metrics

*Human accuracy* — the proportion of human respondents who gave the exact ground-truth target word.
In 62 % of contexts (89/144) no human guessed the target at all; mean accuracy across all
contexts is 0.10 (median 0.00), reaching 1.00 only for the easiest items.

*match\@k* — for a given $k$, the fraction of *unique* human answer types that appear anywhere in
GPT's top-$k$ predictions (unweighted).

*weighted match\@k* — the fraction of *human response mass* covered: for each human answer
found in the top-$k$, its probability (response frequency) is summed.
Because popular answers contribute more, weighted match is consistently higher than unweighted match
and is the more meaningful metric.

= Results

Coverage saturates rapidly: almost all gains occur before $k = 10$, with negligible improvement beyond $k = 20$.

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

In 9 contexts (e.g. _летела_, _миссионеры_, _обгорели_) GPT produces *zero overlap* with
any human answer even at $k = 1000$, suggesting fundamental distributional divergence on
semantically constrained or low-frequency items.

= Summary

GPT-4o-mini's probability mass concentrates on the same words humans prefer most, but its
vocabulary of plausible continuations is much narrower: the long tail of human creativity
(typos aside) is largely absent from the model's distribution.
The sharp saturation of match\@k by $k = 10$ means that simply taking the model's
top-10 candidates is nearly as informative as its entire ranked list.
