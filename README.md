ALBANIAN SEMANTIC SEARCH WITH ZONAL TF-IDF
PROJECT OVERVIEW

This project implements an Information Retrieval (IR) system for Albanian text using:

TF-IDF vectorization

Zonal weighting (Title vs Content)

Custom preprocessing with lemmatization

Cosine similarity ranking

The main objective is to evaluate the impact of preprocessing and lemmatization on semantic search performance in a morphologically rich language like Albanian.

The system compares retrieval results:

With preprocessing (cleaning + lemmatization)

Without preprocessing (raw text)

METHODOLOGY

TEXT PREPARATION (CUSTOM PREPROCESSOR)

Each document’s title and content are merged according to zoning weights (e.g., 0.5–0.5).

A custom preprocessor:

Removes punctuation

Removes digits

Applies lemmatization using Stanza (Albanian model)

@lru_cache is used to optimize repeated lemmatization and improve performance.

FEATURE EXTRACTION (TF-IDF)

Documents are transformed into numerical vectors using TfidfVectorizer.

Main parameters used:

sublinear_tf = True

max_df = 0.85

min_df = 2

These parameters help:

Reduce noise

Downweight very frequent terms

Remove rare, non-informative words

ZONAL WEIGHTING

Three title–content configurations are tested:

0.5 / 0.5 → Balanced

0.3 / 0.7 → Content-heavy

0.7 / 0.3 → Title-heavy

This allows analysis of how structural emphasis affects ranking quality.

SEARCH AND RANKING

User queries are transformed into TF-IDF vectors.

Cosine similarity is used to compare queries with documents.

Top results are retrieved and displayed for:

With preprocessing

Without preprocessing

EVALUATION STRATEGY

Relevance Rating Scale:

Similarity ≥ 0.20 → Good (Strong relevance)

0.10 ≤ Similarity < 0.20 → Fair (Partial relevance)

Similarity < 0.10 → Poor (Weak or no relevance)

Evaluation focuses on:

Ranking stability

Semantic coherence

Contextual accuracy

Morphological handling

ZONAL COMPARISON RESULTS

TITLE 0.5 — CONTENT 0.5 (Balanced)

With preprocessing:

Contextually accurate

Semantically balanced

Clear topic grouping

Without preprocessing:

Keyword repetition

Inconsistent ordering

Occasional mismatches

Impact:
Moderate improvement, strongest in mid-ranked results.

TITLE 0.3 — CONTENT 0.7 (Content-heavy)

With preprocessing:

High semantic precision

Stable clustering

Strong topic cohesion

Without preprocessing:

Fragmented scores

Scattered retrieval

Less focused ranking

Impact:
Strong improvement, especially in deeper ranking positions.

TITLE 0.7 — CONTENT 0.3 (Title-heavy)

With preprocessing:

Conceptually precise

Meaning-based title matching

Without preprocessing:

Keyword-biased ranking

Missed semantic links

Impact:
Noticeable improvement in top-ranked results.

GENERAL FINDINGS

WITH PREPROCESSING:

Consistent rankings across all zoning strategies

Deeper semantic understanding

Improved handling of Albanian morphological variants

Stronger topic coherence

Higher contextual accuracy

WITHOUT PREPROCESSING:

Higher lexical precision

Lower semantic recall

Rankings based on literal matches

Reduced conceptual relevance

FINAL CONCLUSION

Lemmatization significantly enhances meaning-based retrieval in Albanian.

For morphologically rich languages, preprocessing is essential for:

Semantic consistency

Ranking stability

Conceptual relevance

This study demonstrates the measurable advantage of integrating linguistic normalization into classical TF-IDF retrieval pipelines.

TECHNOLOGIES USED

Python

Scikit-learn

Stanza (Albanian NLP)

NumPy

Cosine Similarity

LICENSE

MIT License
