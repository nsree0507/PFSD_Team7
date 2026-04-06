def evaluate_results(results, threshold=0.6):
    """
    Compute evaluation statistics for zero-shot predictions.
    """
    total = len(results)
    accepted = 0
    rejected = 0
    confidence_sum = 0.0

    for res in results:
        confidence_sum += res["confidence"]
        if res["label"] != "Uncertain":
            accepted += 1
        else:
            rejected += 1

    evaluation = {
        "Total_Texts": total,
        "Accepted": accepted,
        "Rejected": rejected,
        "Acceptance_Rate": round(accepted / total, 4),
        "Rejection_Rate": round(rejected / total, 4),
        "Average_Confidence": round(confidence_sum / total, 4)
    }

    return evaluation
