# AI Research Paper Intelligence System

A semantic research assistant built on top of ~15,000 ML papers from arXiv. It started as a straightforward "search papers by meaning" tool and has since grown into a multi-capability research intelligence layer — deep comparative research, live web fallback, entity extraction, citation graphs, and cross-paper trend analysis.

**Core pipeline (unchanged foundation):** Load → Clean → Embed (MiniLM) → Index (FAISS) → Search → Summarize (BART) → Keywords (KeyBERT)
**New intelligence layer (built on top):** Deep Research Mode → Web Search Fallback → NER → Citation Graph → Keyword Intelligence

---

## Why this update

The original version answered one question well: *"which papers are most similar to this query?"* That's useful, but it stops short of what a research assistant actually needs to do — synthesize findings across papers, know when its own dataset doesn't have the answer, pull out structured facts (models, datasets, libraries), show how papers relate to each other, and spot trends across a result set rather than one abstract at a time.

This version doesn't touch the original pipeline — it's still there, cell for cell, doing the same job. Everything new is added as a layer on top that reuses the same `df`, `model`, `index`, `summarizer`, and `kw_model` objects, so nothing that already worked had to be rebuilt.

## What's different: v1 → v2

| Capability | v1 — Semantic Search | v2 — Research Intelligence System |
|---|---|---|
| Core search | Query → top-k papers by cosine similarity | Same, untouched |
| Result quality | Top-k often includes near-duplicate papers on the same sub-topic | Deep Research Mode de-duplicates by topic before synthesizing |
| Data boundary | Silent — if the local 15k papers didn't cover the query, results were just weak matches | Detected automatically (similarity < 0.60) and backfilled with live Semantic Scholar / arXiv results |
| Output format | Raw title + abstract + similarity score | Structured comparative synthesis: common findings, research gaps, future work, best paper |
| Entity awareness | None — abstracts were just blocks of text | NER pulls out models, datasets, libraries, and organizations per paper |
| Relationships between papers | None | Citation graph: locally similar papers + real citations/references via Semantic Scholar |
| Keyword scope | Per single abstract only | Aggregated across a full result set to surface trending topics |
| Summarization | Manual, one abstract at a time | Wired into search and research flows automatically |

## New features — what and why

### 1. Deep Research Mode
Pulls the top 40 candidates instead of 5, removes near-duplicate papers using embedding similarity (so the notebook isn't summarizing five versions of the same paper), then synthesizes the surviving top 20 into common findings, an identified research gap, suggested future work, and a best-paper pick.
**Why:** A single similarity-ranked list doesn't tell you anything about how a field is shaped — you're left reading five abstracts yourself to spot the pattern. This does that synthesis step for you.

### 2. Web Search Integration (Semantic Scholar + arXiv fallback)
If the best local match scores below a 0.60 similarity threshold, the system treats that as "the local dataset doesn't really cover this" and automatically queries the Semantic Scholar Graph API, falling back to arXiv, then merges those results with whatever local matches exist.
**Why:** The local corpus is a fixed 15k-paper snapshot. Without this, a query about something recent or outside that snapshot would just quietly return the closest-but-wrong local paper with no indication it wasn't a real match. This makes that limitation visible and self-correcting instead of silent.

### 3. NER (Named Entity Recognition)
Combines a general-purpose pretrained NER model with hand-built gazetteers for ML-specific terms (model names like BERT/ResNet/GPT-4, datasets like ImageNet/COCO/GLUE, libraries like PyTorch/TensorFlow) to extract structured entities from an abstract.
**Why:** Generic NER models are trained for people/places/organizations — they don't recognize "ResNet" or "GLUE" as meaningful entities. The domain-specific gazetteer fills that gap so the system can answer "which models/datasets does this paper actually use" instead of just returning free text.

### 4. Citation Graph
For any given paper, builds a graph combining locally similar papers (from the existing FAISS index — always available, no external dependency) with real citation/reference data pulled from Semantic Scholar when the paper can be matched online.
**Why:** Similarity search tells you what reads alike; it doesn't tell you what actually cites or builds on what. Combining both gives a more honest picture — similarity as a fallback that's always available, real citation data when it exists.

### 5. Keyword Intelligence
Extends the existing single-abstract KeyBERT extraction to run across an entire search result set, then aggregates keyword frequency to surface trending terms and hot topics for a query rather than just per-paper phrases.
**Why:** Keywords from one abstract tell you about one paper. Aggregating across the top-k results for a query tells you what a sub-field is actually talking about right now — that's a different, more useful question.

## Tech stack

| Layer | Tool | Role |
|---|---|---|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) | Semantic vector representation |
| Vector index | FAISS (`IndexFlatIP`) | Fast local similarity search |
| Summarization | `facebook/bart-large-cnn` | Abstract → concise summary |
| Keyword extraction | KeyBERT | Phrase-level relevance extraction |
| NER | `transformers` NER pipeline + custom ML gazetteers | Structured entity extraction |
| External data | Semantic Scholar Graph API, arXiv API | Live fallback + citation data |
| Graph construction | `networkx` | Citation/similarity graph modeling |

## Setup

```bash
# Core pipeline
pip install datasets sentence-transformers faiss-cpu scikit-learn transformers keybert

# Enhancement layer
pip install requests networkx
```

## Usage

```python
# Original semantic search (unchanged)
search_paper("deep learning for medical image analysis", k=5)
search_and_summarize("deep learning for medical image analysis", k=5)

# Deep Research Mode — synthesized, de-duplicated, multi-paper analysis
deep_research("deep learning for medical image analysis", fetch_k=40, top_k=20)

# Smart search — local FAISS, auto-falls back to live web if local match is weak
smart_search("latest LLM agent benchmarks 2025", k=5)

# NER — extract models/datasets/libraries/orgs from an abstract
extract_entities(df.iloc[10466]["abstract"])

# Citation graph for a specific paper
build_citation_graph(10466, k_similar=5, k_citations=5)

# Keyword Intelligence — trending terms across a result set
keyword_intelligence("deep learning for medical image analysis", k=10)
```

> Run the core pipeline cells first (Steps 1–9) — the enhancement layer reuses `df`, `model`, `index`, `summarizer`, and `kw_model` from those cells and will raise a clear assertion error if they aren't defined yet.

## Project structure

```
├── nlp_project_enhanced.ipynb   # full notebook: core pipeline + intelligence layer
├── paper_embeddings.npy         # cached MiniLM embeddings
├── paper_faiss.index            # cached FAISS index
```

## Honest limitations

- Local corpus is capped at 15,000 papers — Deep Research Mode and Keyword Intelligence are only as good as that subset unless the web fallback kicks in.
- Semantic Scholar / arXiv calls depend on external API availability and aren't cached, so repeated queries re-fetch live.
- NER entity coverage is only as complete as the hand-built gazetteers — new model/dataset names won't be recognized until added.
- `IndexFlatIP` is exact brute-force search; fine at 15k vectors, but would need an approximate index (IVF/HNSW) at larger scale.
