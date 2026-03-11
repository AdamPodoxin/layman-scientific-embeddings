import random
import json
import itertools
from pathlib import Path
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset


DATA_PATH = Path("data")
KEYWORDS_PATH = DATA_PATH / "keywords"

ABSTRACT_JARGON_PAIRS_PATH = DATA_PATH / "pairs" / "abstract-jargon"
ABSTRACT_LAYMAN_PAIRS_PATH = DATA_PATH / "pairs" / "abstract-layman"

TITLE_JARGON_PAIRS_PATH = DATA_PATH / "pairs" / "title-jargon"
TITLE_LAYMAN_PAIRS_PATH = DATA_PATH / "pairs" / "title-layman"

JARGON_JARGON_PAIRS_PATH = DATA_PATH / "pairs" / "jargon-jargon"
LAYMAN_LAYMAN_PAIRS_PATH = DATA_PATH / "pairs" / "layman-layman"
JARGON_LAYMAN_PAIRS_PATH = DATA_PATH / "pairs" / "jargon-layman"

DATASET_PATH = "allenai/scirepeval"
DATASET_NAME = "scidocs_mag_mesh"

# Lower this to only take a subset of pairs
SAMPLE_PROPORTION = 1.0


def create_pairs(list1: list[str], list2: list[str]):
    pairs = list(set(
        (item1, item2)
        for item1 in list1
        for item2 in list2
        if item1 != item2
    ))

    return random.sample(pairs, int(SAMPLE_PROPORTION * len(pairs)))


def get_terms_from_file(path: str):
    with open(path) as file:
        data: dict = json.loads(file.read())
    
    core_entity_terms = data["core_entities"]
    methodology_terms = data["methodologies"]
    outcome_terms = data["outcomes"]

    all_terms = core_entity_terms + methodology_terms + outcome_terms
    jargon_terms: list[str] = list(set(d["jargon"] for d in all_terms))
    layman_terms: list[str] = list(set(d["layman"] for d in all_terms))

    return {
        "jargon": jargon_terms, 
        "layman": layman_terms,
    }


def save_pairs(pairs: list[tuple[str, str]], path: Path | str):
    # df = pd.DataFrame(pairs, columns=["anchor", "positive"])
    # df.to_parquet(path, compression="gzip", index=False)

    pairs_dict = {
        "anchor": [pair[0] for pair in pairs],
        "positive": [pair[1] for pair in pairs],
    }

    ds = Dataset.from_dict(pairs_dict)
    ds_train_test = ds.train_test_split(test_size=0.10)
    ds_test_val = ds_train_test["test"].train_test_split(test_size=0.5)
    ds_train_test_val = DatasetDict({
        "train": ds_train_test["train"],
        "test": ds_test_val["test"],
        "val": ds_test_val["train"],
    })

    ds_train_test_val.save_to_disk(path)


def main():
    file_paths = [path for path in KEYWORDS_PATH.iterdir()]
    doc_ids = [path.stem for path in file_paths]

    terms_per_file = [get_terms_from_file(path) for path in file_paths]
    jargon_terms_per_file = [terms_dict["jargon"] for terms_dict in terms_per_file]
    layman_terms_per_file = [terms_dict["layman"] for terms_dict in terms_per_file]

    ds = load_dataset(DATASET_PATH, DATASET_NAME, split="evaluation")
    id_abstract_dict: dict[str, str | None] = dict(zip(ds["doc_id"], ds["abstract"]))
    id_title_dict: dict[str, str | None] = dict(zip(ds["doc_id"], ds["title"]))

    # Abstracts should not be None because 
    # keyword generation only runs for valid abstracts. 
    abstracts = [id_abstract_dict[id] for id in doc_ids]
    assert all(abstract is not None for abstract in abstracts)

    titles = [id_title_dict[id] for id in doc_ids]

    # Make sure all lists are same length
    assert len(abstracts) == len(titles)
    assert len(abstracts) == len(jargon_terms_per_file)
    assert len(abstracts) == len(layman_terms_per_file)

    abstract_jargon_pairs_per_file = [
        create_pairs([abstract], jargon_terms)
        for abstract, jargon_terms in zip(abstracts, jargon_terms_per_file)
        if abstract is not None
    ]
    abstract_layman_pairs_per_file = [
        create_pairs([abstract], layman_terms)
        for abstract, layman_terms in zip(abstracts, layman_terms_per_file)
        if abstract is not None
    ]

    title_jargon_pairs_per_file = [
        create_pairs([title], jargon_terms)
        for title, jargon_terms in zip(titles, jargon_terms_per_file)
        if title is not None
    ]
    title_layman_pairs_per_file = [
        create_pairs([title], layman_terms)
        for title, layman_terms in zip(titles, layman_terms_per_file)
        if title is not None
    ]

    jargon_jargon_pairs_per_file = [
        create_pairs(jargon_terms, jargon_terms)
        for jargon_terms in jargon_terms_per_file
    ]
    layman_layman_pairs_per_file = [
        create_pairs(layman_terms, layman_terms)
        for layman_terms in layman_terms_per_file
    ]
    jargon_layman_pairs_per_file = [
        create_pairs(jargon_terms, layman_terms)
        for jargon_terms, layman_terms in zip(jargon_terms_per_file, layman_terms_per_file)
    ]

    abstract_jargon_pairs = list(itertools.chain.from_iterable(abstract_jargon_pairs_per_file))
    abstract_layman_pairs = list(itertools.chain.from_iterable(abstract_layman_pairs_per_file))

    title_jargon_pairs = list(itertools.chain.from_iterable(title_jargon_pairs_per_file))
    title_layman_pairs = list(itertools.chain.from_iterable(title_layman_pairs_per_file))
    
    jargon_jargon_pairs = list(itertools.chain.from_iterable(jargon_jargon_pairs_per_file))
    layman_layman_pairs = list(itertools.chain.from_iterable(layman_layman_pairs_per_file))
    jargon_layman_pairs = list(itertools.chain.from_iterable(jargon_layman_pairs_per_file))
    
    save_pairs(abstract_jargon_pairs, ABSTRACT_JARGON_PAIRS_PATH)
    save_pairs(abstract_layman_pairs, ABSTRACT_LAYMAN_PAIRS_PATH)
    
    save_pairs(title_jargon_pairs, TITLE_JARGON_PAIRS_PATH)
    save_pairs(title_layman_pairs, TITLE_LAYMAN_PAIRS_PATH)

    save_pairs(jargon_jargon_pairs, JARGON_JARGON_PAIRS_PATH)
    save_pairs(layman_layman_pairs, LAYMAN_LAYMAN_PAIRS_PATH)
    save_pairs(jargon_layman_pairs, JARGON_LAYMAN_PAIRS_PATH)


if __name__ == "__main__":
    main()
