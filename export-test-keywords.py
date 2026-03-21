import shutil
import json
from pathlib import Path
from itertools import chain
import pandas as pd
from datasets import load_from_disk, concatenate_datasets, Dataset


KEYWORDS_PATH = Path("data") / "keywords"
TEST_KEYWORDS_PATH = Path("data") / "test_keywords"

PAIRS_PATH = Path("data") / "pairs"


def get_keywords_obj(path: Path):
    with open(path) as f:
        return json.loads(f.read())


def get_keywords_set(keywords_obj: dict):
    keyword_lists = list(keywords_obj["core_entities"]) \
                    + list(keywords_obj["methodologies"]) \
                    + list(keywords_obj["outcomes"])
    
    return set(chain.from_iterable(
        [pair["jargon"], pair["layman"]]
        for pair in keyword_lists
    ))


def main():
    pair_datasets = [
        load_from_disk(path)
        for path in PAIRS_PATH.iterdir()
    ]

    test_set: Dataset = concatenate_datasets(
        [ds["test"] for ds in pair_datasets]
    )
    test_keywords = set(test_set["anchor"]) | set(test_set["positive"])

    def are_all_keywords_in_test(keywords: set):
        return all(keyword in test_keywords for keyword in keywords)

    paths = [path for path in KEYWORDS_PATH.iterdir()]

    df = pd.DataFrame(data={
        "path": paths,
        "keywords": [get_keywords_set(get_keywords_obj(path)) for path in paths]
    })

    test_keywords_paths = df[df["keywords"].apply(are_all_keywords_in_test)]["path"]

    for path in test_keywords_paths:
        path: Path = path
        filename = path.name
        shutil.copy(path, TEST_KEYWORDS_PATH / filename)


if __name__ == "__main__":
    main()
