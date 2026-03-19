from pathlib import Path
from sentence_transformers import SentenceTransformer


MODELS_PATH = Path("models")

keywords = [
    "Lithostratigraphy",
    "Layered rock layers",
    "Social production",
]


def print_similarities(model_path: Path):
    model = SentenceTransformer(str(model_path))
    embeddings = model.encode(keywords)
    similarities = model.similarity(embeddings, embeddings)
    print(str(model_path))
    print(similarities)
    print("\n")


def main():
    print_similarities(MODELS_PATH / "vanilla-finetuned")
    print_similarities(MODELS_PATH / "jargon-layman-finetuned 0.25")
    print_similarities(MODELS_PATH / "jargon-layman-finetuned 0.5")
    print_similarities(MODELS_PATH / "jargon-layman-finetuned 0.75")
    print_similarities(MODELS_PATH / "jargon-layman-finetuned 1.0")


if __name__ == "__main__":
    main()