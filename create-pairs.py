import random
import json
import itertools
from pathlib import Path
import pandas as pd


DATA_PATH = Path("data")
KEYWORDS_PATH = DATA_PATH / "keywords"
ALL_PAIRS_PATH = DATA_PATH / "all_pairs.parquet.gzip"
JARGON_LAYMAN_PAIRS_PATH = DATA_PATH / "jargon-layman_pairs.parquet.gzip"

# Lower this to only take a subset of pairs
SAMPLE_PROPORTION = 1.0


def create_pairs(list1: list[str], list2: list[str]):
    pairs = [
        (item1, item2)
        for item1 in list1
        for item2 in list2
        if item1 != item2
    ]

    return random.sample(pairs, int(SAMPLE_PROPORTION * len(pairs)))


def create_pairs_for_file(path: str):
    with open(path) as file:
        data: dict = json.loads(file.read())
    
    core_entities = data["core_entities"]
    methodologies = data["methodologies"]
    outcomes = data["outcomes"]

    jargon_layman_pairs = core_entities + methodologies + outcomes
    jargon_terms = [d["jargon"] for d in jargon_layman_pairs]
    layman_terms = [d["layman"] for d in jargon_layman_pairs]

    jargon_jargon_pairs = set(create_pairs(jargon_terms, jargon_terms))
    layman_layman_pairs = set(create_pairs(layman_terms, layman_terms))
    jargon_layman_pairs = set(create_pairs(jargon_terms, layman_terms))

    all_pairs = set(
        jargon_jargon_pairs
        | layman_layman_pairs
        | jargon_layman_pairs
    )

    return {
        "all_pairs": all_pairs,
        "jargon_layman_pairs": jargon_layman_pairs,
    }


def main():
    pair_results_per_file = [create_pairs_for_file(path) for path in KEYWORDS_PATH.iterdir()]
    all_pairs_per_file = [pair_result["all_pairs"] for pair_result in pair_results_per_file]
    jargon_layman_pairs_per_file = [pair_result["jargon_layman_pairs"] for pair_result in pair_results_per_file]
    
    all_pairs = list(itertools.chain.from_iterable(all_pairs_per_file))

    all_pairs_df = pd.DataFrame(data=set(all_pairs), columns=["anchor", "positive"])
    all_pairs_df.to_parquet(ALL_PAIRS_PATH, compression="gzip", index=False)
    
    jargon_layman_pairs = list(itertools.chain.from_iterable(jargon_layman_pairs_per_file))

    jargon_layman_pairs_df = pd.DataFrame(data=set(jargon_layman_pairs), columns=["anchor", "positive"])
    jargon_layman_pairs_df.to_parquet(JARGON_LAYMAN_PAIRS_PATH, compression="gzip", index=False)


if __name__ == "__main__":
    main()
