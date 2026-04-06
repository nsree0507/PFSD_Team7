from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def generate_dynamic_labels(texts, num_labels=3, top_k_words=3):
    """
    Generate meaningful multi-word hidden labels using clustering + TF-IDF.
    """
    # Step 1: Sentence embeddings
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(texts)

    # Step 2: Clustering
    kmeans = KMeans(n_clusters=num_labels, random_state=42)
    cluster_ids = kmeans.fit_predict(embeddings)

    # Step 3: Group texts by cluster
    clusters = {}
    for idx, cid in enumerate(cluster_ids):
        clusters.setdefault(cid, []).append(texts[idx])

    # Step 4: Extract keyphrases per cluster
    dynamic_labels = []
    for cluster_texts in clusters.values():
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf = vectorizer.fit_transform(cluster_texts)

        feature_names = vectorizer.get_feature_names_out()
        scores = np.asarray(tfidf.sum(axis=0)).flatten()

        top_indices = scores.argsort()[-top_k_words:][::-1]
        keywords = [feature_names[i] for i in top_indices]

        label = " ".join(keywords)
        dynamic_labels.append(label)

    return dynamic_labels
