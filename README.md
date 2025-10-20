[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17402098.svg)](https://doi.org/10.5281/zenodo.17402098)


# Rhetoric Lab (Starter)

A small, reproducible toolkit for analyzing rhetorical patterning in text corpora (think tank reports, websites, PDFs).

## What it does
- Ingest local files and folders (organized by year and source)
- Clean and tokenize text (with optional spaCy lemmatization)
- Compute word and phrase (n-gram) frequencies by year and by source
- Collocations with PMI
- TF–IDF by year (to highlight terms that characterize specific periods)
- Keyword-in-Context (KWIC) concordance
- Export tables (CSV) and simple charts (PNG) for longitudinal analysis

## Folder layout
```
rhetoric_lab_starter/
  data/
    raw/          # put your documents here, e.g., data/raw/Heritage/1995/*.pdf or *.txt
    processed/    # where cleaned text and per-doc metadata will be written
  outputs/
    figures/      # charts
    tables/       # CSV exports
  src/rhetoric_lab/
    *.py          # library modules
  notebooks/
    Starter.ipynb # optional notebook
  examples/
    config.yml    # example config
  requirements.txt
```

## Quick start

1) Create and activate a venv (recommended):
```
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2) Install requirements:
```
pip install -r requirements.txt
```

3) (Optional) Install spaCy English model for better lemmatization:
```
python -m spacy download en_core_web_sm
```

4) Organize your corpus (example structure):
```
data/raw/Heritage/1995/report1.pdf
data/raw/Heritage/1995/report2.pdf
data/raw/AEI/2004/paper.txt
data/raw/Cato/2012/press_release.html
```
TXT and HTML are supported out of the box. PDFs require `pdfminer.six` or `pymupdf` which are included.

5) Run the CLI:
```
python -m rhetoric_lab.cli ingest --input-root data/raw --output-root data/processed
python -m rhetoric_lab.cli analyze --processed-root data/processed --outputs-root outputs --min-count 5 --max-ngram 3
python -m rhetoric_lab.cli kwic --processed-root data/processed --term "personal responsibility" --window 12 --limit 50
```

- The `analyze` step writes CSVs to `outputs/tables` and PNG charts to `outputs/figures`.
- The `kwic` step prints concordance lines and writes a CSV.

## Notes
- If spaCy is installed, the pipeline uses it for tokenization + lemmatization; otherwise it falls back to NLTK.
- You can customize stopwords in `src/rhetoric_lab/stopwords_extra.txt`.
- Euphemism/dog-whistle seeds can be added to `src/rhetoric_lab/seed_categories.yml` and will be summarized in outputs.

## License
MIT (do whatever you want; attribution appreciated).
