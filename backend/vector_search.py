def vector_search(collection, query_embedding, limit=5):
    """
    Perform vector similarity search in MongoDB
    """

    if not query_embedding:
        print("⚠️ Empty embedding received")
        return []

    try:
        pipeline = [
            {
                "$vectorSearch": {
                    "queryVector": query_embedding,
                    "path": "embedding",
                    "numCandidates": 100,
                    "limit": limit,
                    "index": "vector_index"
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "text": 1,
                    "predicted_label": 1,
                    "confidence": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

        results = list(collection.aggregate(pipeline))
        return results

    except Exception as e:
        print("❌ Vector search error:", e)
        return []