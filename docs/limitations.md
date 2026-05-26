# Limitations

This project is an exploratory implementation of hybrid sparse-dense retrieval.

## Current limitations

- Retrieval is currently demonstrated on small examples and sanity checks.
- Score fusion uses a simple weighted combination of sparse and dense scores.
- Systematic hyperparameter tuning for the fusion weight has not yet been performed.
- Model loading depends on Hugging Face availability and local runtime resources.
- Evaluation utilities are present, but benchmark results should be treated as preliminary until run under a fixed experimental setup.

## Planned improvements

- Add more systematic BEIR evaluation runs.
- Compare different fusion strategies.
- Add clearer experiment configuration files.
- Improve logging and result export for reproducibility.