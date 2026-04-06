from classifier.model import load_model
from classifier.predict import (
    predict_static_labels,
    predict_top_k,
    enhanced_prediction   # ⭐ NEW
)
from classifier.dynamic_labels import generate_dynamic_labels
from utils.preprocessing import clean_text
from utils.results_writer import save_results_to_csv
from utils.evaluation import evaluate_results
from utils.visualization import plot_confidence_distribution
from utils.ablation import ablation_study
from utils.explainability import (
    explain_dynamic_labels,
    analyze_uncertain_cases
)

# ⭐ NEW IMPORTS (MongoDB)
from backend.db_connection import predictions_collection
from backend.mongo_queries import insert_prediction
from backend.aggregation import get_dashboard_stats


def main():
    # ---------------------------------
    # Step 1: Input texts
    # ---------------------------------
    texts = [
        "The government announced new economic reforms to control inflation.",
        "The football team won the championship after a thrilling match.",
        "A new AI-powered smartphone was launched with advanced features.",
        "The weather was pleasant throughout the day."
    ]

    # ---------------------------------
    # Step 2: Preprocess texts
    # ---------------------------------
    cleaned_texts = [clean_text(text) for text in texts]

    # ---------------------------------
    # Load zero-shot model ONCE
    # ---------------------------------
    model = load_model()

    # =================================================
    # STEP 1: BASELINE SYSTEM (STATIC LABELS)
    # =================================================
    static_labels = ["Politics", "Economy", "Sports", "Technology"]

    static_results = predict_static_labels(
        classifier=model,
        texts=cleaned_texts,
        labels=static_labels
    )

    print("\n==============================")
    print("BASELINE: STATIC LABEL RESULTS")
    print("==============================")

    for res in static_results:
        print("\nInput Text:")
        print(res["text"])
        print("Predicted Label:", res["label"])
        print("Confidence:", f"{res['confidence']:.4f}")

    # =================================================
    # STEP 1: PROPOSED SYSTEM (DYNAMIC LABELS)
    # =================================================

    dynamic_labels = generate_dynamic_labels(
        cleaned_texts,
        num_labels=3
    )

    print("\n==============================")
    print("AUTO-GENERATED DYNAMIC LABELS")
    print("==============================")
    for lbl in dynamic_labels:
        print("-", lbl)

    dynamic_results_topk = predict_top_k(
        classifier=model,
        texts=cleaned_texts,
        labels=dynamic_labels,
        k=2,
        threshold=0.6
    )

    print("\n==============================")
    print("PROPOSED: DYNAMIC LABEL RESULTS")
    print("==============================")

    formatted_results = []

    for res in dynamic_results_topk:
        print("\nInput Text:")
        print(res["text"])
        print("Top Predictions:")
        for label, score in res["labels"]:
            print(f"  {label}: {score:.4f}")

        best_label, best_score = res["labels"][0]
        formatted_results.append({
            "text": res["text"],
            "label": best_label,
            "confidence": best_score
        })

    # =================================================
    # ⭐ NEW: ADVANCED AI + MONGODB + VECTOR SEARCH
    # =================================================
    print("\n==============================")
    print("ADVANCED: VECTOR + DATABASE RESULTS")
    print("==============================")

    for text in cleaned_texts:
        result = enhanced_prediction(
            classifier=model,
            text=text,
            labels=dynamic_labels,
            collection=predictions_collection
        )

        print("\nText:", result["text"])
        print("Prediction:", result["predicted_label"])
        print("Confidence:", f"{result['confidence']:.4f}")

        print("Similar Cases:")
        for case in result["similar_cases"]:
            print("  ->", case.get("text"))

        # 🔹 Save to MongoDB
        insert_prediction(
            predictions_collection,
            result["text"],
            result["predicted_label"],
            result["confidence"]
        )

    # ---------------------------------
    # Save results to CSV
    # ---------------------------------
    save_results_to_csv(formatted_results)
    print("\nResults saved to results.csv")

    # ---------------------------------
    # Evaluation metrics
    # ---------------------------------
    evaluation = evaluate_results(formatted_results, threshold=0.6)

    print("\n--- Evaluation Summary (Dynamic System) ---")
    for key, value in evaluation.items():
        print(f"{key}: {value}")

    # ---------------------------------
    # Confidence visualization
    # ---------------------------------
    plot_confidence_distribution(formatted_results)

    # =================================================
    # ⭐ NEW: DASHBOARD STATS (FACET)
    # =================================================
    stats = get_dashboard_stats(predictions_collection)

    print("\n==============================")
    print("DASHBOARD STATS (AGGREGATION + FACET)")
    print("==============================")
    print(stats)

    # =================================================
    # STEP 2: ABLATION STUDY (Threshold & K)
    # =================================================
    ablation_results = ablation_study(
        classifier=model,
        texts=cleaned_texts,
        labels=dynamic_labels
    )

    print("\n==============================")
    print("ABLATION STUDY RESULTS")
    print("==============================")

    for res in ablation_results:
        print(
            f"Threshold={res['threshold']}, "
            f"K={res['k']} -> "
            f"Acceptance Rate={res['acceptance_rate']}"
        )

    # =================================================
    # STEP 3A: DYNAMIC LABEL EXPLAINABILITY
    # =================================================
    label_explanations = explain_dynamic_labels(
        cleaned_texts,
        num_labels=3
    )

    print("\n==============================")
    print("DYNAMIC LABEL EXPLANATIONS")
    print("==============================")

    for cluster, info in label_explanations.items():
        print(f"\n{cluster}")
        print("Top Keywords:", ", ".join(info["keywords"]))
        print("Example Texts:")
        for txt in info["example_texts"]:
            print("-", txt)

    # =================================================
    # STEP 3B: FAILURE / UNCERTAIN CASE ANALYSIS
    # =================================================
    uncertain_cases = analyze_uncertain_cases(
        formatted_results,
        threshold=0.6
    )

    print("\n==============================")
    print("UNCERTAIN / FAILURE CASES")
    print("==============================")

    if not uncertain_cases:
        print("No uncertain cases found.")
    else:
        for case in uncertain_cases:
            print("\nText:", case["text"])
            print("Confidence:", f"{case['confidence']:.4f}")
            print("Reason: Low semantic alignment with discovered labels")


if __name__ == "__main__":
    main()