import csv
import os

def save_results_to_csv(results, filename="results.csv"):
    """
    Save prediction results to a CSV file.
    """
    file_exists = os.path.isfile(filename)

    with open(filename, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["Text", "Predicted_Label", "Confidence"]
        )

        if not file_exists:
            writer.writeheader()

        for res in results:
            writer.writerow({
                "Text": res["text"],
                "Predicted_Label": res["label"],
                "Confidence": round(res["confidence"], 4)
            })
