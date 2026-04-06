def get_dashboard_stats(collection):
    """
    Generate dashboard statistics using aggregation + facet
    """
    try:
        pipeline = [
            {
                "$facet": {

                    #  Category distribution
                    "categoryStats": [
                        {
                            "$group": {
                                "_id": "$predicted_label",
                                "count": {"$sum": 1}
                            }
                        },
                        {
                            "$sort": {"count": -1}
                        }
                    ],

                    #  High confidence predictions
                    "highConfidence": [
                        {
                            "$match": {
                                "confidence": {"$gte": 0.8}
                            }
                        },
                        {
                            "$project": {
                                "_id": 0,
                                "text": 1,
                                "predicted_label": 1,
                                "confidence": 1
                            }
                        }
                    ],

                    #  Average confidence per category
                    "avgConfidence": [
                        {
                            "$group": {
                                "_id": "$predicted_label",
                                "avg_score": {"$avg": "$confidence"}
                            }
                        },
                        {
                            "$sort": {"avg_score": -1}
                        }
                    ]
                }
            }
        ]

        result = list(collection.aggregate(pipeline))
        return result

    except Exception as e:
        print("Aggregation error:", e)
        return []