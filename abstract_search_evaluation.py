import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search

from utils import build_layman_document_search_dfs

TOP_K_ABSTRACTS = 5


def get_scores(model: str | Path | SentenceTransformer, top_k_abstracts=TOP_K_ABSTRACTS):
    if not isinstance(model, SentenceTransformer):
        model = SentenceTransformer(str(model))

    query_df, corpus_df = build_layman_document_search_dfs("layman-abstract")

    doc_id_to_corpus_id = {
        doc_id: corpus_id
        for corpus_id, doc_id in enumerate(corpus_df["doc_id"])
    }
    top_k = min(top_k_abstracts, corpus_df.shape[0])

    abstract_embeddings = model.encode_document(corpus_df["document"].to_list())
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
                1 / (rank + 1)
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
