import sys
import json
from pathlib import Path
import pandas as pd
from sentence_transformers.util import semantic_search
from sentence_transformers import SentenceTransformer


TEST_KEYWORDS_PATH = Path("data") / "test_keywords"

NUM_KEYWORDS_PER_ABSTRACT = 15


def read_keywords_file(path: Path) -> dict:
    with open(path) as f:
        return json.loads(f.read())


def get_keywords_from_document(document: dict):
    keyword_pairs: list[dict[str, str]] = list(document["core_entities"]) \
                    + list(document["methodologies"]) \
                    + list(document["outcomes"])
    
    return keyword_pairs


def get_scores(model: str | Path | SentenceTransformer):
    if model is not SentenceTransformer:
        model = SentenceTransformer(str(model))

    df = pd.DataFrame(data={ "path": [path for path in TEST_KEYWORDS_PATH.iterdir()] })
    df["document"] = df["path"].apply(read_keywords_file)
    df["keywords"] = df["document"].apply(get_keywords_from_document)
    df["jargon"] = df["keywords"].apply(lambda keyword_pairs: list(pair["jargon"] for pair in keyword_pairs))
    df["layman"] = df["keywords"].apply(lambda keyword_pairs: list(pair["layman"] for pair in keyword_pairs))

    search_df = df.explode(["jargon", "layman"])[["path", "jargon", "layman"]]\
                    .reset_index()\
                    .drop("index", axis=1)

    jargon_embeddings = model.encode_document(search_df["jargon"])
    layman_embeddings = model.encode_query(search_df["layman"])

    search_results = semantic_search(
        query_embeddings=layman_embeddings,
        corpus_embeddings=jargon_embeddings,
        top_k=NUM_KEYWORDS_PER_ABSTRACT,
    )

    num_keywords = search_df.shape[0]

    perfect_match_score = sum(
        1 if search_results[i][0]["corpus_id"] == i
        else 0
        for i in range(num_keywords)
    ) / num_keywords

    related_keyword_score = sum(
        1 if search_results[i][j]["corpus_id"] == i
        else 0
        for i in range(num_keywords)
        for j in range(NUM_KEYWORDS_PER_ABSTRACT)
    ) / (num_keywords * NUM_KEYWORDS_PER_ABSTRACT)

    return {
        "perfect_match_score": perfect_match_score,
        "related_keyword_score": related_keyword_score,
        "search_results": search_results,
        "search_df": search_df,
    }


def main():
    model_path = sys.argv[1]

    scores = get_scores(model_path)
    print("Full match score:", scores["perfect_match_score"])
    print("Related keywords score:", scores["related_keyword_score"])

if __name__ == "__main__":
    main()
