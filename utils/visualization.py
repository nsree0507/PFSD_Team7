import matplotlib.pyplot as plt

def plot_confidence_distribution(results):
    """
    Plot confidence score distribution.
    """
    confidences = [res["confidence"] for res in results]

    plt.figure(figsize=(7, 5))
    plt.hist(confidences, bins=10)
    plt.xlabel("Confidence Score")
    plt.ylabel("Number of Texts")
    plt.title("Confidence Score Distribution")
    plt.grid(True)
    plt.show()
