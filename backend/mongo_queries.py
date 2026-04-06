from datetime import datetime
from classifier.embedding import generate_embedding


# 🔍 FULL-TEXT SEARCH
def search_text(collection, query):
    """
    Perform keyword-based search using MongoDB text index
    """
    try:
        results = list(collection.find(
            {"$text": {"$search": query}},
            {"score": {"$meta": "textScore"}}
        ).sort([("score", {"$meta": "textScore"})]))

        return results

    except Exception as e:
        print("❌ Full-text search error:", e)
        return []


# 🔗 LOOKUP (JOIN WITH LABELS COLLECTION)
def get_predictions_with_labels(collection):
    """
    Join predictions with labels collection to get descriptions
    """
    try:
        pipeline = [
            {
                "$lookup": {
                    "from": "labels",
                    "localField": "predicted_label",
                    "foreignField": "label_name",
                    "as": "label_info"
                }
            },
            {
                "$unwind": {
                    "path": "$label_info",
                    "preserveNullAndEmptyArrays": True
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "text": 1,
                    "predicted_label": 1,
                    "confidence": 1,
                    "description": "$label_info.description"
                }
            }
        ]

        return list(collection.aggregate(pipeline))

    except Exception as e:
        print("❌ Lookup error:", e)
        return []


# 📥 INSERT PREDICTION WITH EMBEDDING
def insert_prediction(collection, text, label, confidence):
    """
    Insert prediction along with embedding into MongoDB
    """
    try:
        embedding = generate_embedding(text)

        document = {
            "text": text,
            "embedding": embedding,
            "predicted_label": label,
            "confidence": confidence,
            "timestamp": datetime.utcnow()
        }

        collection.insert_one(document)

    except Exception as e:
        print("❌ Insert error:", e)