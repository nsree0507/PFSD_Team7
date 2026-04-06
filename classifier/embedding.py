from sentence_transformers import SentenceTransformer

#  Load model once (important for performance)
model = SentenceTransformer('all-MiniLM-L6-v2')


def generate_embedding(text: str):
    """
    Convert input text into vector embedding
    """
    if not text or not text.strip():
        return []

    try:
        embedding = model.encode(text)
        return embedding.tolist()
    except Exception as e:
        print("❌ Embedding error:", e)
        return []