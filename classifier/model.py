from transformers import pipeline

def load_model():
    """
    Load pretrained zero-shot classification model.
    """
    classifier = pipeline(
        task="zero-shot-classification",
        model="facebook/bart-large-mnli"
    )
    return classifier
