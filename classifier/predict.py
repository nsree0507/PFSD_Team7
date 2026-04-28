# ==========================================
# IMPORTS
# ==========================================
from transformers import pipeline
from classifier.embedding import generate_embedding
from vector_search import vector_search
from db_connection import tickets_collection

# ==========================================
# LOAD MODEL (runs once)
# ==========================================
classifier = pipeline("zero-shot-classification")

# Default ERP labels
DEFAULT_LABELS = [
    "Fees Issue",
    "Attendance Issue",
    "Marks Issue",
    "Exam Query",
    "Hostel Issue",
    "Library Issue",
    "General Query"
]

# ==========================================
# BASELINE: Static label prediction
# ==========================================
def predict_static_labels(texts, labels=DEFAULT_LABELS):
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
def predict_top_k(texts, labels=DEFAULT_LABELS, k=2, threshold=0.6):
    results = []

    for text in texts:
        output = classifier(text, labels)
        scores = dict(zip(output["labels"], output["scores"]))

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
# ENHANCED PREDICTION (AI + VECTOR SEARCH)
# ==========================================
def enhanced_prediction(query, collection=None):
    """
    Predict intent category from user query
    """

    result = classifier(query, DEFAULT_LABELS)

    top_label = result["labels"][0]
    confidence = round(result["scores"][0], 2)

    return {
        "query": query,
        "category": top_label,
        "confidence": confidence
    }

# ==========================================
# MAIN FUNCTION FOR FLASK (IMPORTANT)
# ==========================================
def predict(text):
    """
    This is the function Flask will call
    """

    result = enhanced_prediction(text)

    return result["predicted_label"]
