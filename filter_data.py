"""Shared data-loading helpers that filter to Russian-only tokens.

Every analysis script should use load_human() and load_gpt() instead of
raw pd.read_csv() to ensure consistent filtering across the pipeline.
"""

import pandas as pd


def load_human(path: str = "people_with_prob.csv") -> pd.DataFrame:
    """Load human cloze data, keep only Russian answers, renormalize probability_y."""
    df = pd.read_csv(path)
    df = df[df["is_russian"] == True].copy()  # noqa: E712

    # Recompute probability_y: within each context, the proportion of
    # respondents who gave each answer (among remaining Russian answers).
    df["probability_y"] = df.groupby("word.id")["answer"].transform(
        lambda s: s.map(s.value_counts(normalize=True))
    )
    return df


def load_gpt(path: str = "gpt4omini_morph_2.csv") -> pd.DataFrame:
    """Load GPT-4o-mini predictions, keep only Russian tokens."""
    df = pd.read_csv(path)
    return df[df["is_russian"] == True].copy()  # noqa: E712
