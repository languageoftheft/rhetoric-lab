from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_term_trends(tables_root: str, term: str, outputs_root: str, source: str=None, ngram: bool=False) -> str:
    tables_root = Path(tables_root)
    outputs_root = Path(outputs_root)
    outputs_root.mkdir(parents=True, exist_ok=True)

    if ngram:
        df = pd.read_csv(tables_root / "ngram_frequencies_by_source_year.csv")
        df = df[df["ngram"].str.lower() == term.lower()]
        label_col = "ngram"
    else:
        df = pd.read_csv(tables_root / "word_frequencies_by_source_year.csv")
        df = df[df["term"].str.lower() == term.lower()]
        label_col = "term"

    if source:
        df = df[df["source"] == source]

    if df.empty:
        return ""

    grouped = df.groupby(["year"], as_index=False)["count"].sum().sort_values("year")
    plt.figure()
    plt.plot(grouped["year"], grouped["count"], marker="o")
    plt.title(f"Trend for '{term}'")
    plt.xlabel("Year")
    plt.ylabel("Count")
    out_path = outputs_root / f"trend_{term.replace(' ','_')}.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return str(out_path)
