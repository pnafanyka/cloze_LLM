== POS Probability Correlation

Code: `analysis_pos_corr.py` · Data: `output/pos_correlation.csv`, `output/pos_delta.csv`

=== Method

While the POS overlap analysis (preceding section) asks whether humans and GPT-4o-mini
_use_ the same parts of speech, it does not test whether the _probability mass_ assigned
to each POS category is similar. Two complementary algorithms address this question.

*Setup.* For each of the 144 contexts, a POS probability vector is constructed for both
humans and the model. Human responses are first deduplicated by surface form within each
context, taking the first recorded cloze probability and UPOS tag for every unique answer;
the probabilities are then summed by POS tag. Model predictions are deduplicated by
`prediction_cleaned`, retaining the highest probability for each unique surface form, and
likewise summed by POS. Missing POS categories receive a probability of zero.

*Algorithm 2 (per-POS correlation).* Each POS tag defines a pair of 144-element vectors —
one from humans, one from the model — containing the summed probability mass that POS
received in each context. Pearson and Spearman correlations are computed for each POS tag,
restricted to tags that appear in at least 10 contexts.

*Algorithm 3 (delta table).* For each context, the cell-wise absolute difference
$delta_c ["POS"] = |p_"human" (c, "POS") - p_"model" (c, "POS")|$ is computed for every
POS tag. The mean delta for a context is the average of these absolute differences across
all POS categories; the overall mean delta is the grand average across all 144 contexts.
A mean delta of zero would indicate perfect distributional alignment.

=== Results

Algorithm 2 yields the following per-POS correlations (sorted by Pearson $r$, descending):

#align(center)[
#table(
  columns: (auto, auto, auto, auto, auto),
  align: (center, center, center, center, center),
  table.header[POS][Pearson $r$][Pearson $p$][Spearman $r$][Spearman $p$],
  [NUM],   [0.94], [$< 0.001$], [0.58], [$< 0.001$],
  [ADP],   [0.84], [$< 0.001$], [0.40], [$< 0.001$],
  [SCONJ], [0.76], [$< 0.001$], [0.40], [$< 0.001$],
  [CCONJ], [0.71], [$< 0.001$], [0.47], [$< 0.001$],
  [NOUN],  [0.70], [$< 0.001$], [0.81], [$< 0.001$],
  [VERB],  [0.55], [$< 0.001$], [0.60], [$< 0.001$],
  [ADJ],   [0.52], [$< 0.001$], [0.39], [$< 0.001$],
  [ADV],   [0.43], [$< 0.001$], [0.33], [$< 0.001$],
  [PRON],  [0.34], [$< 0.001$], [0.49], [$< 0.001$],
  [PART],  [0.30], [$< 0.001$], [0.30], [$< 0.001$],
  [DET],   [0.26], [$= 0.001$], [0.40], [$< 0.001$],
  [X],     [0.02], [$= 0.776$], [0.05], [$= 0.513$],
  [INTJ],  [$-0.004$], [$= 0.965$], [0.03], [$= 0.760$],
  [PROPN], [$-0.007$], [$= 0.938$], [0.21], [$= 0.012$],
  [PUNCT], [$-0.04$], [$= 0.644$], [0.01], [$= 0.861$],
)
]

Algorithm 3 yields an overall mean delta of *0.046* across all 144 contexts (range:
0.0002 -- 0.112).

=== Interpretation

The per-POS correlations reveal a clear hierarchy. Closed-class functional categories that
are syntactically constrained — NUM ($r = 0.94$), ADP ($r = 0.84$), SCONJ ($r = 0.76$),
CCONJ ($r = 0.71$) — exhibit the strongest agreement between human cloze responses and
model predictions. These categories are heavily determined by the preceding syntactic
context, and both humans and GPT-4o-mini respond similarly to these constraints.

Open-class content words show moderate agreement: NOUN ($r = 0.70$), VERB ($r = 0.55$),
ADJ ($r = 0.52$). The relatively high Spearman correlation for NOUN ($rho = 0.81$)
suggests strong rank-order agreement even when absolute probability magnitudes differ.

The weakest correlations are observed for categories that are either rare or
context-insensitive: PUNCT ($r = -0.04$), PROPN ($r = -0.01$), INTJ ($r = -0.004$), and
X ($r = 0.02$). The model's tendency to assign substantial probability to punctuation
tokens (which rarely appear in human cloze responses) drives the near-zero or negative
PUNCT correlation.

The overall mean delta of 0.046 from Algorithm 3 indicates that, on average, each POS
category's probability differs between humans and the model by about 4.6 percentage
points. While this confirms that human and model POS distributions are broadly aligned
at the category level, the non-trivial per-context range (up to 0.112) shows that
alignment varies considerably across sentence contexts.
