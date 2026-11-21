"""Blind hand-labelling of news headlines for an in-domain evaluation set.

Usage (from the project root, in a normal terminal)::

    python -m src.pipeline sample-headlines        # creates data/labelled/<ticker>_headlines_gold.csv
    python -m src.labelling                        # label interactively; safe to quit and resume

Keys: n = negative, u = neutral, p = positive, s = skip (unclear), b = back, q = save and quit.
The file can also be filled in a spreadsheet: put negative / neutral / positive / skip in
the ``label`` column. Guidelines: docs/annotation_guidelines.md.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import DATA_DIR, DEFAULT_TICKER, LABELS, RANDOM_SEED

LABELLED_DIR = DATA_DIR / "labelled"
VALID_LABELS = set(LABELS) | {"skip"}
KEYS = {"n": "negative", "u": "neutral", "p": "positive", "s": "skip"}


def gold_path(ticker: str = DEFAULT_TICKER, labelled_dir: Path = LABELLED_DIR) -> Path:
    return labelled_dir / f"{ticker.lower()}_headlines_gold.csv"


def sample_for_annotation(news: pd.DataFrame, n: int = 300, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Random sample stratified by year, shuffled, with only the fields an annotator needs.

    No model output is included, so labels cannot be anchored on predictions.
    """
    news = news.copy()
    news["Date"] = pd.to_datetime(news["Date"], utc=True)
    n = min(n, len(news))
    years = news["Date"].dt.year
    shares = years.value_counts(normalize=True)
    per_year = (shares * n).round().astype(int)
    per_year.iloc[0] += n - per_year.sum()  # make the total exactly n

    parts = [
        news[years == year].sample(n=min(k, int((years == year).sum())), random_state=seed)
        for year, k in per_year.items() if k > 0
    ]
    sample = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out = sample[["Date", "Article_title"]].copy()
    out.insert(0, "item_id", range(1, len(out) + 1))
    out["label"] = ""
    return out


def load_gold(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"label": "string"}, keep_default_na=False)
    df["label"] = df["label"].fillna("").str.strip().str.lower()
    bad = sorted(set(df["label"]) - VALID_LABELS - {""})
    if bad:
        raise ValueError(f"{path}: unknown labels {bad}; use {sorted(VALID_LABELS)}")
    return df


def labelled_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a usable gold label (skips and blanks removed)."""
    return df[df["label"].isin(LABELS)].copy()


def progress_line(df: pd.DataFrame) -> str:
    counts = df["label"].replace("", "unlabelled").value_counts()
    done = int(df["label"].isin(VALID_LABELS).sum())
    detail = ", ".join(f"{k}={counts.get(k, 0)}" for k in [*LABELS, "skip"])
    return f"{done}/{len(df)} done ({detail})"


def interactive_label(path: Path, input_fn=input, print_fn=print) -> pd.DataFrame:
    df = load_gold(path)
    todo = [i for i in df.index if df.at[i, "label"] == ""]
    print_fn(__doc__.split("Keys:")[1].split("The file")[0].strip())
    print_fn(progress_line(df))

    history = []
    pos = 0
    while pos < len(todo):
        i = todo[pos]
        print_fn(f"\n[{df.at[i, 'item_id']}] {str(df.at[i, 'Date'])[:10]}  {df.at[i, 'Article_title']}")
        key = input_fn("label [n/u/p/s, b=back, q=quit]: ").strip().lower()
        if key == "q":
            break
        if key == "b":
            if history:
                pos = history.pop()
                df.at[todo[pos], "label"] = ""
            continue
        if key not in KEYS:
            print_fn("  use n, u, p, s, b or q")
            continue
        df.at[i, "label"] = KEYS[key]
        df.to_csv(path, index=False)  # save after every label
        history.append(pos)
        pos += 1

    df.to_csv(path, index=False)
    print_fn("\n" + progress_line(df))
    return df


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.labelling", description="Label headlines for evaluation")
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--file", type=Path, default=None,
                        help="gold CSV (default data/labelled/<ticker>_headlines_gold.csv)")
    args = parser.parse_args(argv)

    path = args.file or gold_path(args.ticker)
    if not path.exists():
        print(f"{path} not found - run: python -m src.pipeline sample-headlines", file=sys.stderr)
        return 1
    interactive_label(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())


def sample_training_headlines(
    news: pd.DataFrame,
    n: int = 400,
    exclude_tickers: tuple = (DEFAULT_TICKER,),
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Relevant headlines from *other* companies, spread evenly over tickers, for headline fine-tuning.

    Excluding the evaluation ticker keeps its annotated headlines a clean, company-disjoint test set.
    """
    from src.relevance import add_relevance

    pool = add_relevance(news)
    pool = pool[pool["relevant"] & ~pool["Stock_symbol"].isin(exclude_tickers)]
    held_out_titles = set(news.loc[news["Stock_symbol"].isin(exclude_tickers), "Article_title"])
    pool = pool[~pool["Article_title"].isin(held_out_titles)]  # same headline tagged to a held-out ticker
    pool = pool.drop_duplicates(subset="Article_title")
    per_ticker = max(1, -(-n // pool["Stock_symbol"].nunique()))  # ceil
    shuffled = pool.sample(frac=1.0, random_state=seed)
    sample = shuffled.groupby("Stock_symbol").head(per_ticker).head(n).reset_index(drop=True)
    out = sample[["Date", "Stock_symbol", "Article_title"]].copy()
    out.insert(0, "item_id", range(1, len(out) + 1))
    out["label"] = ""
    return out
