# AGENTS.md

This repository is intentionally small and optimized for learning how a basic RAG application works.

## Principles

- Keep the code easy to read in one sitting.
- Prefer explicit code over abstractions.
- Do not add extra routes beyond `/health`, `/index`, and `/ask`.
- Keep the RAG lifecycle visible: loading, chunking, embedding, indexing, retrieval, and answer generation.

## File Guide

- `app/main.py`: FastAPI app and routes.
- `app/config.py`: environment-driven settings.
- `app/rag_pipeline.py`: the full RAG flow.
- `app/schemas.py`: request and response models.
- `sample_docs/`: local documents to index.

## Working Style

- If you change the stack or flow, update `README.md` so the teaching explanation stays accurate.
- Favor minimal dependencies and clear comments.
