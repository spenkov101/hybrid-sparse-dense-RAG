# Reproducibility Notes

This project is exploratory and intended for incremental research development.

## Current setup

The retriever components use:

- SPLADE for sparse lexical retrieval
- Contriever for dense semantic retrieval
- simple hybrid score fusion for ranking

## Notes

- Model downloads depend on Hugging Face availability.
- Retrieval outputs may vary slightly across environments due to model/runtime versions.
- BEIR-style evaluation utilities are included for future systematic comparison.
- Notebooks are intended for sanity checks and qualitative inspection, not final benchmark reporting.