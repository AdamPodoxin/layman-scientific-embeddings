import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search

from utils import build_keyword_search_df

NUM_KEYWORDS_PER_ABSTRACT = 15


def get_scores(model: str | Path | SentenceTransformer):
    if type(model) is not SentenceTransformer:
        model = SentenceTransformer(str(model))

    search_df = build_keyword_search_df()

    unique_jargon = search_df["jargon"].drop_duplicates().reset_index(drop=True)
    jargon_to_corpus_id = {
        jargon: corpus_id
        for corpus_id, jargon in enumerate(unique_jargon)
    }

    jargon_embeddings = model.encode_document(unique_jargon.to_list())
    layman_embeddings = model.encode_query(search_df["layman"].to_list())

    top_k = min(NUM_KEYWORDS_PER_ABSTRACT, len(unique_jargon))
    gold_corpus_ids = search_df["jargon"].map(jargon_to_corpus_id).to_list()

    search_results = semantic_search(
        query_embeddings=layman_embeddings,
        corpus_embeddings=jargon_embeddings,
        top_k=top_k,
    )

    num_keywords = search_df.shape[0]

    perfect_match_score = sum(
        1 if search_results[i][0]["corpus_id"] == gold_corpus_ids[i]
        else 0
        for i in range(num_keywords)
    ) / num_keywords

    mean_reciprocal_rank_at_15 = sum(
        next(
            (
                1 / (rank + 1)
                for rank, result in enumerate(search_results[i])
                if result["corpus_id"] == gold_corpus_ids[i]
            ),
            0,
        )
        for i in range(num_keywords)
    ) / num_keywords

    related_keyword_score = sum(
        1 if search_results[i][j]["corpus_id"] == gold_corpus_ids[i]
        else 0
        for i in range(num_keywords)
        for j in range(top_k)
    ) / (num_keywords * top_k)

    jargon_to_doc_ids = search_df.groupby("jargon")["doc_id"].apply(set).to_dict()

    recall_at_15 = sum(
        1 if any(
            search_df.iloc[i]["doc_id"] in jargon_to_doc_ids.get(unique_jargon.iloc[result["corpus_id"]], set())
            for result in search_results[i][:top_k]
        )
        else 0
        for i in range(num_keywords)
    ) / num_keywords

    precision_at_15 = sum(
        1 if search_df.iloc[i]["doc_id"] in jargon_to_doc_ids.get(
            unique_jargon.iloc[search_results[i][j]["corpus_id"]], set()
        )
        else 0
        for i in range(num_keywords)
        for j in range(top_k)
    ) / (num_keywords * top_k)

    f1_at_15 = (
        2 * (precision_at_15 * recall_at_15) / (precision_at_15 + recall_at_15)
        if (precision_at_15 + recall_at_15) > 0
        else 0
    )

    return {
        "perfect_match_score": perfect_match_score,
        "mean_reciprocal_rank_at_15": mean_reciprocal_rank_at_15,
        "related_keyword_score": related_keyword_score,
        "recall_at_15": recall_at_15,
        "precision_at_15": precision_at_15,
        "f1_at_15": f1_at_15,
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
