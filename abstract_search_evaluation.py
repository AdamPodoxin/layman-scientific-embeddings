import sys
import json
from pathlib import Path
import pandas as pd
from sentence_transformers.util import semantic_search
from sentence_transformers import SentenceTransformer
from datasets import load_dataset


TEST_KEYWORDS_PATH = Path("data") / "test_keywords"

TOP_K_ABSTRACTS = 5


def read_keywords_file(path: Path) -> dict:
    with open(path) as f:
        return json.loads(f.read())


def get_layman_keywords_from_document(document: dict):
    keyword_pairs: list[dict[str, str]] = list(document["core_entities"]) \
                    + list(document["methodologies"]) \
                    + list(document["outcomes"])
    
    return [pair["layman"] for pair in keyword_pairs]


def get_scores(model: str | Path | SentenceTransformer, top_k_abstracts=TOP_K_ABSTRACTS):
    if not isinstance(model, SentenceTransformer):
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
    corpus_df = merged_df[["doc_id", "abstract"]] \
                    .drop_duplicates(subset="doc_id") \
                    .reset_index(drop=True)
    query_df = merged_df[["doc_id", "layman"]] \
                    .explode("layman") \
                    .reset_index(drop=True)

    doc_id_to_corpus_id = {
        doc_id: corpus_id
        for corpus_id, doc_id in enumerate(corpus_df["doc_id"])
    }
    top_k = min(top_k_abstracts, corpus_df.shape[0])

    abstract_embeddings = model.encode_document(corpus_df["abstract"].to_list())
    layman_embeddings = model.encode_query(query_df["layman"].to_list())

    search_results = semantic_search(
        query_embeddings=layman_embeddings,
        corpus_embeddings=abstract_embeddings,
        top_k=top_k,
    )

    target_corpus_ids = query_df["doc_id"].map(doc_id_to_corpus_id).to_list()
    num_keywords = query_df.shape[0]

    perfect_match_score = sum(
        1 if search_results[i][0]["corpus_id"] == target_corpus_ids[i]
        else 0
        for i in range(num_keywords)
    ) / num_keywords

    mean_reciprocal_rank_at_5 = sum(
        next(
            (
                1 / (rank + 1)  # rank is 0-indexed, so rank + 1 gives the 1-indexed rank
                for rank, result in enumerate(search_results[i])
                if result["corpus_id"] == target_corpus_ids[i]
            ),
            0,
        )
        for i in range(num_keywords)
    ) / num_keywords

    recall_at_5 = sum(
        1 if any(result["corpus_id"] == target_corpus_ids[i] for result in search_results[i])
        else 0
        for i in range(num_keywords)
    ) / num_keywords

    return {
        "perfect_match_score": perfect_match_score,
        "mean_reciprocal_rank_at_5": mean_reciprocal_rank_at_5,
        "recall_at_5": recall_at_5,
        "search_results": search_results,
        "search_df": query_df,
        "corpus_df": corpus_df,
    }


def main():
    model_path = sys.argv[1]

    scores = get_scores(model_path)
    print("Full match score:", scores["perfect_match_score"])
    print("Mean reciprocal rank@5:", scores["mean_reciprocal_rank_at_5"])
    print("Recall@5:", scores["recall_at_5"])

if __name__ == "__main__":
    main()
