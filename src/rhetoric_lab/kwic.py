import json
from pathlib import Path
from typing import List
import pandas as pd

def _kwic_lines(text: str, term: str, window: int) -> List[dict]:
    text_l = text.lower()
    term_l = term.lower()
    idx = 0
    out = []
    while True:
        i = text_l.find(term_l, idx)
        if i == -1:
            break
        start = max(0, i - window)
        end = min(len(text), i + len(term) + window)
        out.append({"left": text[start:i], "term": text[i:i+len(term)], "right": text[i+len(term):end]})
        idx = i + len(term)
    return out

def kwic_concordance(processed_root: str, term: str, window: int=12, limit: int=100, out_csv: str=None) -> pd.DataFrame:
    processed_root = Path(processed_root)
    rows = []
    for txt in processed_root.rglob("*.txt"):
        text = txt.read_text(errors="ignore")
        meta = json.loads(txt.with_suffix(".meta.json").read_text())
        lines = _kwic_lines(text, term, window)
        for ln in lines[:max(0, limit - len(rows))]:
            rows.append({
                "source": meta.get("source","unknown"),
                "year": meta.get("year","unknown"),
                "left": ln["left"],
                "term": ln["term"],
                "right": ln["right"],
                "doc": str(txt)
            })
        if len(rows) >= limit:
            break
    df = pd.DataFrame(rows)
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df
