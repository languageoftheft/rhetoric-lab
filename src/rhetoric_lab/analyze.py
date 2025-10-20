from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import json
import math

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def _load_docs(processed_root: str) -> List[Dict]:
    processed_root = Path(processed_root)
    docs = []
    for tok in processed_root.rglob("*.tokens.json"):
        tokens = json.loads(tok.read_text())
        meta = json.loads(tok.with_suffix(".meta.json").read_text())
        docs.append({"tokens": tokens, "meta": meta, "path": str(tok)})
    return docs

def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def pmi(coll_counts: Counter, word_counts: Counter, total_tokens: int) -> Dict[Tuple[str,str], float]:
    # pointwise mutual information for bigrams only
    pmi_vals = {}
    for (w1,w2), cxy in coll_counts.items():
        px = word_counts[w1] / total_tokens
        py = word_counts[w2] / total_tokens
        pxy = cxy / total_tokens
        if px > 0 and py > 0 and pxy > 0:
            pmi_vals[(w1,w2)] = math.log2(pxy / (px*py))
    return pmi_vals

class Analyzer:
    def __init__(self, processed_root: str, outputs_root: str, min_count: int=5, max_ngram: int=3):
        self.processed_root = Path(processed_root)
        self.outputs_root = Path(outputs_root)
        self.min_count = min_count
        self.max_ngram = max_ngram
        (self.outputs_root / "tables").mkdir(parents=True, exist_ok=True)
        (self.outputs_root / "figures").mkdir(parents=True, exist_ok=True)

    def build_tables(self) -> Dict[str, pd.DataFrame]:
        docs = _load_docs(str(self.processed_root))
        # group tokens by (source, year)
        by_group = defaultdict(list)
        for d in docs:
            src = d["meta"].get("source","unknown")
            yr = d["meta"].get("year","unknown")
            by_group[(src, yr)].append(d["tokens"])

        rows_words = []
        rows_ngrams = []
        rows_colloc = []
        rows_tfidf = []

        # For TF-IDF by year across entire corpus
        texts_by_year = defaultdict(list)

        for (src, yr), doc_tokens_list in by_group.items():
            all_tokens = [t for doc in doc_tokens_list for t in doc]
            total_tokens = len(all_tokens)
            wc = Counter(all_tokens)

            # words
            for w, c in wc.items():
                if c >= self.min_count:
                    rows_words.append({"source": src, "year": yr, "term": w, "count": c, "total_tokens": total_tokens})

            # n-grams
            for n in range(2, self.max_ngram+1):
                ngr = Counter([" ".join(ng) for doc in doc_tokens_list for ng in _ngrams(doc, n)])
                for k, c in ngr.items():
                    if c >= self.min_count:
                        rows_ngrams.append({"source": src, "year": yr, "ngram": k, "n": n, "count": c, "total_tokens": total_tokens})

            # bigram collocations with PMI
            bigrams = Counter([ng for doc in doc_tokens_list for ng in _ngrams(doc, 2)])
            pmi_vals = pmi(bigrams, wc, max(total_tokens,1))
            for (w1,w2), val in pmi_vals.items():
                c = bigrams[(w1,w2)]
                if c >= self.min_count:
                    rows_colloc.append({"source": src, "year": yr, "w1": w1, "w2": w2, "count": c, "pmi": val})

            # TF-IDF by year prep
            texts_by_year[yr].append(" ".join(all_tokens))

        # Build TF-IDF
        years = sorted(texts_by_year.keys())
        corpus = [" ".join(texts_by_year[y]) for y in years]
        if len(corpus) >= 1:
            vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1,2))
            X = vectorizer.fit_transform(corpus)
            terms = vectorizer.get_feature_names_out()
            import numpy as np
            for i, yr in enumerate(years):
                row = X[i].toarray().ravel()
                top_idx = np.argsort(-row)[:200]
                for j in top_idx:
                    rows_tfidf.append({"year": yr, "term": terms[j], "tfidf": float(row[j])})

        df_words = pd.DataFrame(rows_words).sort_values(["source","year","count"], ascending=[True,True,False])
        df_ngrams = pd.DataFrame(rows_ngrams).sort_values(["source","year","count"], ascending=[True,True,False])
        df_colloc = pd.DataFrame(rows_colloc).sort_values(["source","year","pmi"], ascending=[True,True,False])
        df_tfidf = pd.DataFrame(rows_tfidf).sort_values(["year","tfidf"], ascending=[True,False])

        # Save
        out_tables = self.outputs_root / "tables"
        df_words.to_csv(out_tables / "word_frequencies_by_source_year.csv", index=False)
        df_ngrams.to_csv(out_tables / "ngram_frequencies_by_source_year.csv", index=False)
        df_colloc.to_csv(out_tables / "bigram_collocations_pmi_by_source_year.csv", index=False)
        df_tfidf.to_csv(out_tables / "tfidf_top_terms_by_year.csv", index=False)

        return {
            "words": df_words,
            "ngrams": df_ngrams,
            "colloc": df_colloc,
            "tfidf": df_tfidf
        }
