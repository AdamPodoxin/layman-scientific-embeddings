import sys
import json
from pathlib import Path
import pandas as pd
from sentence_transformers.util import semantic_search
from sentence_transformers import SentenceTransformer
from datasets import Dataset, load_dataset


TEST_KEYWORDS_PATH = Path("data") / "test_keywords"

TOP_K_TITLES = 5


def read_keywords_file(path: Path) -> dict:
    with open(path) as f:
        return json.loads(f.read())


def get_layman_keywords_from_document(document: dict):
    keyword_pairs: list[dict[str, str]] = list(document["core_entities"]) \
                    + list(document["methodologies"]) \
                    + list(document["outcomes"])
    
    return [pair["layman"] for pair in keyword_pairs]


def get_scores(model: str | Path | SentenceTransformer, top_k_titles=TOP_K_TITLES):
    if type(model) is not SentenceTransformer:
        model = SentenceTransformer(str(model))

    paths = [path for path in TEST_KEYWORDS_PATH.iterdir()]
    ids = [path.stem for path in paths]

    df = pd.DataFrame(data={ 
        "path": paths,
        "doc_id": ids,
    })
    df["document"] = df["path"].apply(read_keywords_file)
    df["layman"] = df["document"].apply(get_layman_keywords_from_document)

    dataset_df = load_dataset("allenai/scirepeval", "scidocs_mag_mesh", split="evaluation").to_pandas()
    merged_df = pd.merge(df, dataset_df, on="doc_id")
    search_df = merged_df.explode("layman")[["layman", "title"]] \
                            .reset_index() \
                            .drop("index", axis=1)

    title_embeddings = model.encode_document(search_df["title"].to_list())
    layman_embeddings = model.encode_query(search_df["layman"].to_list())

    search_results = semantic_search(
        query_embeddings=layman_embeddings,
        corpus_embeddings=title_embeddings,
        top_k=top_k_titles,
    )

    num_keywords = search_df.shape[0]

    perfect_match_score = sum(
        1 if search_results[i][0]["corpus_id"] == i
        else 0
        for i in range(num_keywords)
    ) / num_keywords

    in_top_results_score = sum(
        1 if any(search_results[i][j]["corpus_id"] == i for j in range(TOP_K_TITLES))
        else 0
        for i in range(num_keywords)
    ) / num_keywords

    return {
        "perfect_match_score": perfect_match_score,
        "in_top_results_score": in_top_results_score,
        "search_results": search_results,
        "search_df": search_df,
    }


def main():
    model_path = sys.argv[1]

    scores = get_scores(model_path)
    print("Full match score:", scores["perfect_match_score"])
    print("In top results score:", scores["in_top_results_score"])

if __name__ == "__main__":
    main()
