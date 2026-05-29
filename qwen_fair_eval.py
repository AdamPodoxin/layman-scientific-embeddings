"""Fair Qwen evaluation across precision and checkpoint formats."""
from pathlib import Path

from sentence_transformers import SentenceTransformer

from abstract_search_evaluation import get_scores as get_abstract_scores
from keyword_search_evaluation import get_scores as get_keyword_scores
from title_search_evaluation import get_scores as get_title_scores
from utils import (
    QWEN_MODEL_ID,
    load_finetuned_qwen,
    load_qwen_base_4bit,
    load_qwen_base_bf16,
    round_down_to_2_decimals,
)

MODELS_PATH = Path("models")


def load_models() -> dict[str, SentenceTransformer]:
    models = {
        "base_fp16": SentenceTransformer(QWEN_MODEL_ID, device="cuda"),
        "base_4bit": load_qwen_base_4bit(),
        "base_bf16": load_qwen_base_bf16(),
        "vanilla_lora": load_finetuned_qwen(MODELS_PATH / "vanilla-qwen"),
        "layman_jargon_lora": load_finetuned_qwen(MODELS_PATH / "layman-jargon-qwen"),
    }

    vanilla_merged = MODELS_PATH / "vanilla-qwen-merged"
    layman_merged = MODELS_PATH / "layman-jargon-qwen-merged"
    if vanilla_merged.exists():
        models["vanilla_merged"] = load_finetuned_qwen(vanilla_merged)
    if layman_merged.exists():
        models["layman_jargon_merged"] = load_finetuned_qwen(layman_merged)

    return models


def print_keyword_scores(name: str, model: SentenceTransformer) -> None:
    scores = get_keyword_scores(model)
    print(
        f"{name}: MRR@15={round_down_to_2_decimals(scores['mean_reciprocal_rank_at_15'])} "
        f"Recall@15={round_down_to_2_decimals(scores['recall_at_15'])} "
        f"Precision@15={round_down_to_2_decimals(scores['precision_at_15'])}"
    )


def print_title_scores(name: str, model: SentenceTransformer) -> None:
    scores = get_title_scores(model)
    print(
        f"{name}: MRR={round_down_to_2_decimals(scores['mean_reciprocal_rank_at_5'])} "
        f"Recall@5={round_down_to_2_decimals(scores['recall_at_5'])}"
    )


def print_abstract_scores(name: str, model: SentenceTransformer) -> None:
    scores = get_abstract_scores(model)
    print(
        f"{name}: MRR={round_down_to_2_decimals(scores['mean_reciprocal_rank_at_5'])} "
        f"Recall@5={round_down_to_2_decimals(scores['recall_at_5'])}"
    )


def main() -> None:
    models = load_models()

    print("=== Keyword search (layman -> jargon) ===")
    for name, model in models.items():
        print_keyword_scores(name, model)

    print("\n=== Title search (layman keywords -> titles) ===")
    for name, model in models.items():
        print_title_scores(name, model)

    print("\n=== Abstract search (layman keywords -> abstracts) ===")
    for name, model in models.items():
        print_abstract_scores(name, model)


if __name__ == "__main__":
    main()
