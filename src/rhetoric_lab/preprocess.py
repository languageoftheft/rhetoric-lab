import re
from pathlib import Path
from typing import Iterable, List, Optional
from collections import Counter
import json

# Optional spaCy; fallback to NLTK
try:
    import spacy
    _SPACY_OK = True
except Exception:
    _SPACY_OK = False

import nltk
from nltk.corpus import stopwords as _nltk_stopwords
from nltk import word_tokenize
try:
    _ = _nltk_stopwords.words("english")
except Exception:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)

class TextPreprocessor:
    def __init__(self, use_spacy: bool=True, lemmatize: bool=True, extra_stopwords_path: Optional[str]=None):
        self.use_spacy = use_spacy and _SPACY_OK
        self.lemmatize = lemmatize
        self.stopwords = set(_nltk_stopwords.words("english"))
        if extra_stopwords_path:
            p = Path(extra_stopwords_path)
            if p.exists():
                for line in p.read_text().splitlines():
                    line = line.strip().lower()
                    if not line or line.startswith("#"):
                        continue
                    self.stopwords.add(line)

        if self.use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "textcat"])
            except Exception:
                self.use_spacy = False

    def normalize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        # keep words and apostrophes
        tokens = word_tokenize(text)
        toks = []
        if self.use_spacy:
            doc = self.nlp(" ".join(tokens))
            for t in doc:
                if not t.text.strip():
                    continue
                if t.is_stop:
                    continue
                if self.lemmatize:
                    lemma = t.lemma_.strip().lower()
                else:
                    lemma = t.text.strip().lower()
                if len(lemma) < 2:
                    continue
                if lemma in self.stopwords:
                    continue
                toks.append(lemma)
        else:
            for t in tokens:
                t = t.strip().lower()
                if len(t) < 2:
                    continue
                if t in self.stopwords:
                    continue
                toks.append(t)
        return toks

    def save_tokens(self, processed_root: str) -> None:
        processed_root = Path(processed_root)
        for txt in processed_root.rglob("*.txt"):
            text = txt.read_text(errors="ignore")
            tokens = self.normalize(text)
            meta_path = txt.with_suffix(".meta.json")
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
            else:
                meta = {}
            meta["tokens"] = len(tokens)
            meta_path.write_text(json.dumps(meta, indent=2))
            tok_path = txt.with_suffix(".tokens.json")
            tok_path.write_text(json.dumps(tokens))
