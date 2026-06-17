import argparse
import random
import json
from pathlib import Path
from datasets import Dataset, DatasetDict


def normalize_keywords(keywords: dict) -> dict:
    """Ensure each category is a list of {jargon, layman} dicts (Arrow-compatible)."""
    normalized = {}
    for cat in ("core_entities", "methodologies", "outcomes"):
        items = keywords.get(cat, [])
        if not isinstance(items, list):
            items = []
        pairs = []
        for item in items:
            if isinstance(item, dict) and "jargon" in item and "layman" in item:
                pairs.append({"jargon": str(item["jargon"]), "layman": str(item["layman"])})
            elif isinstance(item, str):
                pairs.append({"jargon": item, "layman": item})
        normalized[cat] = pairs
    return normalized


def read_keywords_file(path: Path):
    with open(path) as f:
        raw = json.loads(f.read())
    if not isinstance(raw, dict):
        raise ValueError(f"Expected object in {path}, got {type(raw).__name__}")
    return normalize_keywords(raw)


def parse_args():
    p = argparse.ArgumentParser("Split keywords into train, val, and test sets")

    p.add_argument(
        "--keywords-path",
        type=Path,
        default=Path("data") / "keywords",
        help="Path to keywords directory",
    )
    p.add_argument(
        "--output-path",
        type=Path,
        default=Path("data") / "keywords_split",
        help="Path to output directory",
    )
    p.add_argument(
        "--test-ratio",
        type=float,
        default=0.05,
        help="Ratio of keywords to put in test set",
    )
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Ratio of keywords to put in validation set",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    keyword_path = Path(args.keywords_path)
    output_path = Path(args.output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    file_paths = [path for path in keyword_path.iterdir()]
    random.seed(args.seed)
    random.shuffle(file_paths)

    num_docs = len(file_paths)
    num_test = int(num_docs * float(args.test_ratio))
    num_val = int(num_docs * float(args.val_ratio))
    num_train = num_docs - num_test - num_val

    test_paths = file_paths[:num_test]
    val_paths = file_paths[num_test:num_test+num_val]
    train_paths = file_paths[num_test+num_val:]

    test_doc_ids = [path.stem for path in test_paths]
    val_doc_ids = [path.stem for path in val_paths]
    train_doc_ids = [path.stem for path in train_paths]

    test_ds = Dataset.from_dict({
        "doc_id": test_doc_ids,
        "keywords": [read_keywords_file(path) for path in test_paths],
    })
    val_ds = Dataset.from_dict({
        "doc_id": val_doc_ids,
        "keywords": [read_keywords_file(path) for path in val_paths],
    })
    train_ds = Dataset.from_dict({
        "doc_id": train_doc_ids,
        "keywords": [read_keywords_file(path) for path in train_paths],
    })

    ds_dict = DatasetDict({
        "test": test_ds,
        "val": val_ds,
        "train": train_ds,
    })

    ds_dict.save_to_disk(output_path)
