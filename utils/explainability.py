from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np


def explain_dynamic_labels(texts, num_labels=3, top_k_words=5):
    """
    Explain how dynamic labels are formed using keywords and example texts.
    """
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(texts)

    kmeans = KMeans(n_clusters=num_labels, random_state=42)
    cluster_ids = kmeans.fit_predict(embeddings)

    explanations = {}

    for cluster_id in range(num_labels):
        cluster_texts = [
            texts[i] for i in range(len(texts)) if cluster_ids[i] == cluster_id
        ]

        if not cluster_texts:
            continue

        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf = vectorizer.fit_transform(cluster_texts)

        feature_names = vectorizer.get_feature_names_out()
        scores = np.asarray(tfidf.sum(axis=0)).flatten()

        top_indices = scores.argsort()[-top_k_words:][::-1]
        keywords = [feature_names[i] for i in top_indices]

        explanations[f"Cluster_{cluster_id}"] = {
            "keywords": keywords,
            "example_texts": cluster_texts
        }

    return explanations
def analyze_uncertain_cases(results, threshold=0.6):
    """
    Analyze texts classified as Uncertain.
    """
    uncertain_cases = []

    for res in results:
        if res["label"] == "Uncertain" or res["confidence"] < threshold:
            uncertain_cases.append({
                "text": res["text"],
                "confidence": res["confidence"]
            })

    return uncertain_cases
