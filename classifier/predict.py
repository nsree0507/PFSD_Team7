# ==========================================
# IMPORTS (NEW + REQUIRED)
# ==========================================
from classifier.embedding import generate_embedding
from backend.vector_search import vector_search


# ==========================================
# BASELINE: Static label prediction
# ==========================================
def predict_static_labels(classifier, texts, labels):
    """
    Baseline zero-shot classification using static (user-defined) labels.
    """
    results = []

    for text in texts:
        output = classifier(text, labels)
        scores = dict(zip(output["labels"], output["scores"]))

        best_label = max(scores, key=scores.get)
        best_score = scores[best_label]

        results.append({
            "text": text,
            "label": best_label,
            "confidence": best_score
        })

    return results


# ==========================================
# PROPOSED: Top-K dynamic label prediction
# ==========================================
def predict_top_k(classifier, texts, labels, k=2, threshold=0.6):
    """
    Perform zero-shot classification with Top-K label selection.
    """
    results = []

    for text in texts:
        output = classifier(text, labels)
        scores = dict(zip(output["labels"], output["scores"]))

        # Sort by confidence
        sorted_scores = sorted(
            scores.items(), key=lambda x: x[1], reverse=True
        )

        top_k = sorted_scores[:k]

        if top_k[0][1] < threshold:
            results.append({
                "text": text,
                "labels": [("Uncertain", top_k[0][1])]
            })
        else:
            results.append({
                "text": text,
                "labels": top_k
            })

    return results


# ==========================================
# ⭐ NEW: ENHANCED PREDICTION (AI + MONGODB)
# ==========================================
def enhanced_prediction(classifier, text, labels, collection):
    """
    Advanced prediction with:
    - Embedding generation
    - Vector search (semantic similarity)
    - Zero-shot classification
    """

    # 🔹 Step 1: Generate embedding
    embedding = generate_embedding(text)

    # 🔹 Step 2: Vector search (similar past cases)
    similar_cases = vector_search(collection, embedding)

    # 🔹 Step 3: Zero-shot classification
    output = classifier(text, labels)
    scores = dict(zip(output["labels"], output["scores"]))

    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]

    return {
        "text": text,
        "embedding": embedding,
        "predicted_label": best_label,
        "confidence": best_score,
        "similar_cases": similar_cases
    }