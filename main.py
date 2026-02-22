import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import stanza
import re
from tqdm import tqdm

# ==========================
# 1. Load Dataset
# ==========================
df = pd.read_csv("001-news-items.csv")
df = df.dropna(subset=["title", "content"])
print(f"Loaded {len(df)} news articles.")

# ==========================
# 2. Initialize Stanza (run once)
# ==========================
stanza.download("sq", verbose=False)
nlp = stanza.Pipeline("sq", processors="tokenize,lemma", use_gpu=False, verbose=False)

# ==========================
# 3. Faster Lemmatizer (cached)
# ==========================
from functools import lru_cache

@lru_cache(maxsize=50000)
def stanza_lemmatize_cached(text):
    doc = nlp(text)
    lemmas = [w.lemma for s in doc.sentences for w in s.words]
    return " ".join(lemmas)

# ==========================
# 4. Zonal Weighting Function
# ==========================
def weighted_text(title, content, w_title, w_content):
    # Instead of repeating strings, multiply token counts
    return ((title + " ") * int(w_title * 10)) + ((content + " ") * int(w_content * 10))

# ==========================
# 5. Preprocessing (optional)
# ==========================
def custom_preprocessor(text):
    text = re.sub(r"\d+", "", text.lower())  # remove numbers
    text = re.sub(r"[^\w\s]", "", text)      # remove punctuation
    return stanza_lemmatize_cached(text)

# ==========================
# 6. Build TF-IDF
# ==========================
def build_tfidf(df, w_title, w_content, use_lemmatizer=False):
    docs = [weighted_text(t, c, w_title, w_content) for t, c in zip(df["title"], df["content"])]

    if use_lemmatizer:
        vectorizer = TfidfVectorizer(
            preprocessor=custom_preprocessor,
            sublinear_tf=True, max_df=0.85, min_df=2
        )
    else:
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.85, min_df=2)

    X = vectorizer.fit_transform(docs)
    return vectorizer, X

# ==========================
# 7. Retrieve Top Documents
# ==========================
def retrieve_top_docs(query, vectorizer, X, df, top_n=5, use_lemmatizer=False):
    if use_lemmatizer:
        query_vec = vectorizer.transform([custom_preprocessor(query)])
    else:
        query_vec = vectorizer.transform([query])
    cosine_sim = cosine_similarity(query_vec, X).flatten()
    ranked_indices = cosine_sim.argsort()[::-1][:top_n]
    return [
        {
            "title": df.iloc[idx]["title"],
            "similarity": cosine_sim[idx],
            "snippet": df.iloc[idx]["content"][:250] + "..."
        }
        for idx in ranked_indices
    ]

# ==========================
# 8. Run Experiments
# ==========================
weights = [(0.5, 0.5), (0.3, 0.7), (0.7, 0.3)]
query = "dhoma e ulet e parlamentit"

for w_title, w_content in weights:
    print(f"\n==============================")
    print(f"ZONAL WEIGHTS -> Title: {w_title}, Content: {w_content}")
    print("==============================")

    vectorizer, X = build_tfidf(df, w_title, w_content, use_lemmatizer=True)
    results = retrieve_top_docs(query, vectorizer, X, df, top_n=5, use_lemmatizer=True)

    for i, r in enumerate(results, 1):
        print(f"\nRank {i} | Similarity: {r['similarity']:.4f}")
        print(f"Title: {r['title']}")
        print(f"Snippet: {r['snippet']}")

# ==========================
# 9. Comparison Without Lemmatization
# ==========================
print("\n=====================================")
print("COMPARISON: Without Lemmatization")
print("=====================================")

vectorizer_plain, X_plain = build_tfidf(df, 0.5, 0.5, use_lemmatizer=False)
results_plain = retrieve_top_docs(query, vectorizer_plain, X_plain, df, top_n=5)

for i, r in enumerate(results_plain, 1):
    print(f"\nRank {i} | Similarity: {r['similarity']:.4f}")
    print(f"Title: {r['title']}")
    print(f"Snippet: {r['snippet']}")
