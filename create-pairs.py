import json
import itertools
from pathlib import Path
import pandas as pd


DATA_PATH = Path("data")
KEYWORDS_PATH = DATA_PATH / "keywords"
PAIRS_PATH = DATA_PATH / "all_pairs.parquet.gzip"


def create_pairs(list1: list[str], list2: list[str]):
    return [
        (item1, item2)
        for item1 in list1
        for item2 in list2
        if item1 != item2
    ]


def create_pairs_for_file(path: str):
    with open(path) as file:
        data: dict = json.loads(file.read())
    
    core_entities = data["core_entities"]
    methodologies = data["methodologies"]
    outcomes = data["outcomes"]

    jargon_layman_pairs = core_entities + methodologies + outcomes
    jargon_terms = [d["jargon"] for d in jargon_layman_pairs]
    layman_terms = [d["layman"] for d in jargon_layman_pairs]

    return set(
        create_pairs(jargon_terms, jargon_terms)
        + create_pairs(layman_terms, layman_terms)
        + create_pairs(jargon_terms, layman_terms)
    )


def main():
    pairs_per_file = [create_pairs_for_file(path) for path in KEYWORDS_PATH.iterdir()]
    all_pairs = list(itertools.chain.from_iterable(pairs_per_file))

    all_pairs_df = pd.DataFrame(data=set(all_pairs), columns=["anchor", "positive"])
    all_pairs_df.to_parquet(PAIRS_PATH, compression="gzip", index=False)


if __name__ == "__main__":
    main()
