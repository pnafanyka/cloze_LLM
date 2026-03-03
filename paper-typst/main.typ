#align(center)[
  #text(17pt)[*Human vs GPT-4o-mini Cloze Predictions*]

  #v(6pt)
  #datetime.today().display()
]

#v(10pt)

= Introduction

Predictability is a central variable in psycholinguistic models of reading and language
comprehension. It is typically operationalized as the conditional probability of a word given
its preceding context and is most commonly estimated using the cloze procedure (Taylor, 1953).
In this paradigm, participants are presented with sentence fragments and asked to provide a
continuation; predictability is computed as the proportion of respondents who produce the
target word.

Extensive research has demonstrated that predictability systematically modulates online
processing. Highly predictable words are associated with shorter reading times, reduced
fixation durations, and lower processing costs relative to less predictable words.
Predictability effects have been observed across multiple dependent measures, including
first fixation duration, gaze duration, total reading time, and regression probability.
These effects are generally interpreted within probabilistic or expectation-based frameworks
of comprehension, according to which readers continuously generate and update probabilistic
expectations about upcoming linguistic input.

Importantly, predictability does not operate in isolation. It interacts with other lexical
and contextual variables, including word frequency, word length, morphological complexity,
and syntactic structure. Although frequency and predictability are often correlated, they
reflect distinct sources of information: frequency indexes general lexical familiarity,
whereas predictability reflects context-specific expectation. Empirical studies have shown
that both variables independently contribute to variance in eye-movement measures, supporting
the view that readers integrate global lexical statistics with local contextual constraints
during processing.

= Related Work

Jacobs et al. (2024) conducted a series of experiments explicitly designed to compare
language model–generated continuations with human cloze norms collected on the same task.
They performed a large-scale evaluation of cloze completion data and demonstrated that
token prediction tasks are neither lexically nor semantically aligned with human responses.
Although language models often retrieve plausible continuations, their rank correspondence
with human cloze probabilities reveals systematic distortions: frequent human responses are
under-ranked, whereas rare responses are over-ranked. Even the most probable human responses
are retrieved in first position by models only in a minority of cases. Furthermore, analyses
of semantic embedding spaces indicate that human and model-generated continuations tend to
occupy partially distinct regions, suggesting distributional misalignment at the semantic level.

Ilia and Aziz (2024), using the Provo Corpus, compared full conditional probability
distributions produced by humans and several large language models. Their results indicate
that humans exhibit graded uncertainty in next-word prediction, whereas model distributions
are substantially narrower and more concentrated. In terms of total variation distance,
model-generated conditional probability distributions diverge systematically from human
distributions, reflecting overconfidence and reduced dispersion.

In contrast, Rego, Snell, and Meeter (2024) demonstrated that large language model–derived
predictability estimates may outperform traditional cloze probabilities in explaining
eye-movement measures within a cognitive model of reading. Using autoregressive
transformer-based models, they showed that model-based surprisal improved model fit across
multiple eye-tracking metrics. Notably, LLaMA-based probability estimates reduced prediction
error more effectively than cloze-based estimates in the OB1-reader framework.

Taken together, these findings raise a critical methodological question: can large language
models serve as reliable substitutes for human-derived predictability norms? The evidence
suggests that models may approximate certain aspects of processing difficulty, yet they
diverge from human distributions at the lexical and semantic levels.

= Research Question

A central question underlying this work is whether large language models can, in principle,
serve as substitutes for human participants in the estimation of predictability norms. If
such substitution is possible, it is necessary to determine under what conditions it may be
justified and which aspects of human expectations are adequately captured by computational
models.

The motivation for this inquiry is both theoretical and practical. From a methodological
perspective, the collection of psycholinguistic norms requires substantial time, financial
resources, and coordinated participant recruitment. Large-scale norming studies involve
extensive human labor and are therefore costly to conduct and difficult to replicate across
languages and domains. If language models could approximate human probability distributions
with sufficient fidelity, they might provide a scalable alternative for norm generation.

However, the issue is not merely one of accuracy in next-word prediction. As emphasized in
prior work, including Language Models for Cloze Task Answer Generation in Russian (2020),
the crucial question concerns the extent to which model-derived probability distributions
align with human expectations at different representational levels (lexical, syntactic, or
semantic). A model may correctly retrieve the target word, yet distribute probability mass
in a manner that diverges substantially from human response patterns.

In the present study, we therefore seek to evaluate this question within the framework of
an experimentally validated dataset. Rather than constructing a new artificial benchmark,
we assess model behavior using a cloze test that has already been administered to human
participants. This design allows for a direct comparison between model-generated probability
distributions and empirically observed human norms under identical contextual constraints.

= Task and Data

A *cloze task* presents a sentence with the final word removed; respondents supply what
comes next.
The dataset covers *144 Russian sentence contexts*.
Human responses were collected from 628 subjects (14–151 per context, avg 48),
yielding 6 898 individual responses with 1–65 unique answers per context.
GPT-4o-mini predictions were obtained as a ranked list of token continuations
with log-probabilities, converted to probabilities via $exp(sum "logprob")$.
After deduplication by stripped surface form, each context has between 6 and 13 957 distinct
GPT candidates.

= Metrics

*Human accuracy* — the proportion of human respondents who gave the exact ground-truth
target word.
In 62 % of contexts (89/144) no human guessed the target at all; mean accuracy across all
contexts is 0.10 (median 0.00), reaching 1.00 only for the easiest items.

*match\@k* — for a given $k$, the fraction of *unique* human answer types that appear
anywhere in GPT's top-$k$ predictions (unweighted).

*weighted match\@k* — the fraction of *human response mass* covered: for each human answer
found in the top-$k$, its probability (response frequency) is summed.
Because popular answers contribute more, weighted match is consistently higher than
unweighted match and is the more meaningful metric.

= Results

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

In 9 contexts (e.g. _летела_, _миссионеры_, _обгорели_) GPT produces *zero overlap* with
any human answer even at $k = 1000$, suggesting fundamental distributional divergence on
semantically constrained or low-frequency items.

= Summary

GPT-4o-mini's probability mass concentrates on the same words humans prefer most, but its
vocabulary of plausible continuations is much narrower: the long tail of human creativity
(typos aside) is largely absent from the model's distribution.
The sharp saturation of match\@k by $k = 10$ means that simply taking the model's
top-10 candidates is nearly as informative as its entire ranked list.

= New Analyses

#include "section_lexical.typ"
#include "section_pos_overlap.typ"
#include "section_pos_corr.typ"
#include "section_entropy.typ"
#include "section_target.typ"
