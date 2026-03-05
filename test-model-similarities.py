from pathlib import Path
from sentence_transformers import SentenceTransformer


MODELS_PATH = Path("models")
VANILLA_FINETUNED_MODEL_PATH = MODELS_PATH / "vanilla-finetuned"
JARGON_LAYMAN_FINETUNED_MODEL_PATH = MODELS_PATH / "jargon-layman-finetuned"


def main():
    keywords = [
        "Lithostratigraphy",
        "Layered rock layers",
        "Social production",
    ]

    print("Vanilla finetuned model:")
    model = SentenceTransformer(str(VANILLA_FINETUNED_MODEL_PATH))

    embeddings = model.encode(keywords)
    print(embeddings)

    similarities = model.similarity(embeddings, embeddings)
    print(similarities)

    print("")
    print("Jargon-Layman finetuned model")
    model = SentenceTransformer(str(JARGON_LAYMAN_FINETUNED_MODEL_PATH))

    embeddings = model.encode(keywords)
    print(embeddings)

    similarities = model.similarity(embeddings, embeddings)
    print(similarities)


if __name__ == "__main__":
    main()