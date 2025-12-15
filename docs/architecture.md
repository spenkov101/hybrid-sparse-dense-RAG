# Hybrid Sparse–Dense RAG Architecture

This document describes the high-level architecture of the hybrid retrieval system
implemented in this repository.

## Overview

The pipeline combines **sparse lexical matching** (SPLADE) and **dense semantic matching**
(Contriever), followed by a weighted hybrid scoring step.

![Hybrid Sparse–Dense RAG Architecture](architecture.png)

## Retrieval Flow

1. **User Query**  
   Natural language query provided by the user.

2. **SPLADE Sparse Encoding**  
   Captures exact term matches and informative rare tokens.

3. **Contriever Dense Encoding**  
   Captures semantic similarity and paraphrastic matches.

4. **Hybrid Scoring**  
   Sparse and dense scores are combined using a weighting factor α.

5. **Ranked Results**  
   Passages are returned sorted by final hybrid score.

## Notes

- Retrieval-only system (no generation step).
- Evaluation is performed using BEIR-style metrics via `ir_measures`.
