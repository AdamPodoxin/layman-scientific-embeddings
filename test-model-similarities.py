from pathlib import Path
from sentence_transformers import SentenceTransformer


MODEL_PATH = Path("models") / "scideberta-full-jargon-layman-keywords"


def main():
    keywords = [
        "AI-based detection methods",
        "system structure",
        "foam nest",
    ]

    model = SentenceTransformer(str(MODEL_PATH))

    embeddings = model.encode(keywords)
    print(embeddings)

    similarities = model.similarity(embeddings, embeddings)
    print(similarities)


if __name__ == "__main__":
    main()