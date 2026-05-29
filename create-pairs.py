import argparse
import random
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


DATASET_PATH = "allenai/scirepeval"
DATASET_NAME = "scidocs_mag_mesh"

SPLITS = ("train", "val", "test")

PAIR_TYPES = (
    "jargon-abstract",
    "layman-abstract",
    "jargon-title",
    "layman-title",
    "jargon-jargon",
    "layman-layman",
    "layman-jargon",
)


def parse_args():
    p = argparse.ArgumentParser("Create training pairs from split keywords")

    p.add_argument(
        "--keywords-split-path",
        type=Path,
        default=Path("data") / "keywords_split",
        help="Path to keywords_split DatasetDict",
    )
    p.add_argument(
        "--output-path",
        type=Path,
        default=Path("data") / "pairs",
        help="Path to output pairs DatasetDict",
    )
    p.add_argument(
        "--sample-proportion",
        type=float,
        default=1.0,
        help="Proportion of pairs to sample per document",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return p.parse_args()


def create_pairs(
    list1: list[str],
    list2: list[str],
    sample_proportion: float,
) -> list[tuple[str, str]]:
    pairs = list(set(
        (item1, item2)
        for item1 in list1
        for item2 in list2
        if item1 != item2
    ))

    if not pairs:
        return []

    return random.sample(pairs, int(sample_proportion * len(pairs)))


def get_terms_from_keywords(keywords: dict) -> dict[str, list[str]]:
    all_keywords = (
        keywords["core_entities"]
        + keywords["methodologies"]
        + keywords["outcomes"]
    )
    return {
        "jargon": list({d["jargon"] for d in all_keywords}),
        "layman": list({d["layman"] for d in all_keywords}),
    }


def get_pairs_to_records(
    pairs: list[tuple[str, str]],
    doc_id: str,
    pair_type: str,
) -> list[dict[str, str]]:
    return [
        {
            "anchor": anchor,
            "positive": positive,
            "doc_id": doc_id,
            "pair_type": pair_type,
        }
        for anchor, positive in pairs
    ]


def get_pairs_to_record(records: list[dict[str, str]]) -> Dataset:
    return Dataset.from_dict({
        "anchor": [record["anchor"] for record in records],
        "positive": [record["positive"] for record in records],
        "doc_id": [record["doc_id"] for record in records],
        "pair_type": [record["pair_type"] for record in records],
    })


def build_pairs_for_split(
    split_ds: Dataset,
    id_abstract_dict: dict[str, str | None],
    id_title_dict: dict[str, str | None],
    sample_proportion: float,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []

    for doc_id, keywords in zip(split_ds["doc_id"], split_ds["keywords"]):
        terms = get_terms_from_keywords(keywords)
        jargon_terms = terms["jargon"]
        layman_terms = terms["layman"]

        abstract = id_abstract_dict[doc_id]
        title = id_title_dict[doc_id]

        if abstract is not None:
            records.extend(get_pairs_to_records(
                create_pairs(jargon_terms, [abstract], sample_proportion),
                doc_id,
                "jargon-abstract",
            ))
            records.extend(get_pairs_to_records(
                create_pairs(layman_terms, [abstract], sample_proportion),
                doc_id,
                "layman-abstract",
            ))

        if title is not None:
            records.extend(get_pairs_to_records(
                create_pairs(jargon_terms, [title], sample_proportion),
                doc_id,
                "jargon-title",
            ))
            records.extend(get_pairs_to_records(
                create_pairs(layman_terms, [title], sample_proportion),
                doc_id,
                "layman-title",
            ))

        records.extend(get_pairs_to_records(
            create_pairs(jargon_terms, jargon_terms, sample_proportion),
            doc_id,
            "jargon-jargon",
        ))
        records.extend(get_pairs_to_records(
            create_pairs(layman_terms, layman_terms, sample_proportion),
            doc_id,
            "layman-layman",
        ))
        records.extend(get_pairs_to_records(
            create_pairs(layman_terms, jargon_terms, sample_proportion),
            doc_id,
            "layman-jargon",
        ))

    return records


def main():
    args = parse_args()
    random.seed(args.seed)

    keywords_split: DatasetDict = load_from_disk(str(args.keywords_split_path))

    ds = load_dataset(DATASET_PATH, DATASET_NAME, split="evaluation")
    id_abstract_dict: dict[str, str | None] = dict(zip(ds["doc_id"], ds["abstract"]))
    id_title_dict: dict[str, str | None] = dict(zip(ds["doc_id"], ds["title"]))

    pairs_dict = DatasetDict({
        split: get_pairs_to_record(build_pairs_for_split(
            keywords_split[split],
            id_abstract_dict,
            id_title_dict,
            args.sample_proportion,
        ))
        for split in SPLITS
    })

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    pairs_dict.save_to_disk(args.output_path)


if __name__ == "__main__":
    main()
