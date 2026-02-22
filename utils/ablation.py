def ablation_study(classifier, texts, labels):
    """
    Perform ablation study on threshold and Top-K values.
    """
    thresholds = [0.4, 0.6, 0.8]
    k_values = [1, 2, 3]

    results = []

    for t in thresholds:
        for k in k_values:
            accepted = 0

            for text in texts:
                output = classifier(text, labels)
                scores = dict(zip(output["labels"], output["scores"]))

                sorted_scores = sorted(
                    scores.items(), key=lambda x: x[1], reverse=True
                )

                best_score = sorted_scores[0][1]

                if best_score >= t:
                    accepted += 1

            results.append({
                "threshold": t,
                "k": k,
                "accepted": accepted,
                "total": len(texts),
                "acceptance_rate": round(accepted / len(texts), 4)
            })

    return results
