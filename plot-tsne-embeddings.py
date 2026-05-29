import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils import (
    build_keyword_search_df,
    build_layman_document_search_dfs,
    load_finetuned_qwen,
)


def parse_args():
    p = argparse.ArgumentParser("Plot t-SNE model embeddings")

    p.add_argument(
        "--model-path",
        type=str,
        default="models/vanilla-scibert",
        help="The HF model id or local directory",
    )
    p.add_argument(
        "--lora-qwen",
        action="store_true",
        help="If the model is a finetuned Qwen model, then use the loading util",
    )
    p.add_argument(
        "--model-name",
        type=str,
        default="vanilla-scibert",
        help="Model name to display in title",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("plots/tsne/vanilla-scibert.png"),
        help="Output image path",
    )
    p.add_argument(
        "--num-docs",
        type=int,
        default=10,
        help="Number of documents whose keywords to plot",
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if bool(args.lora_qwen):
        model = load_finetuned_qwen(args.model_path)
    else:
        model = SentenceTransformer(args.model_path, device="cuda")

    keyword_df = build_keyword_search_df()
    _, abstract_corpus_df = build_layman_document_search_dfs("layman-abstract")

    keyword_df = keyword_df.rename(columns={"doc_id": "id"})
    abstract_corpus_df = abstract_corpus_df.rename(columns={"doc_id": "id", "document": "abstract"})

    ids_to_plot = sorted(keyword_df["id"].unique())[: int(args.num_docs)]

    df_filtered = keyword_df[keyword_df["id"].isin(ids_to_plot)]
    abstract_df_filtered = abstract_corpus_df[abstract_corpus_df["id"].isin(ids_to_plot)]

    jargon_embeddings = model.encode(df_filtered["jargon"].to_list())
    layman_embeddings = model.encode(df_filtered["layman"].to_list())
    abstract_embeddings = model.encode(abstract_df_filtered["abstract"].to_list())

    # Combine all embeddings for consistent dimensionality reduction
    all_embeddings = np.vstack([jargon_embeddings, layman_embeddings, abstract_embeddings])
    
    # First reducing to 50 components using PCA
    # before running through t-SNE, as per docs.
    pca = PCA(n_components=50)
    all_pca = pca.fit_transform(all_embeddings)
    
    # Split back into separate arrays
    jargon_pca = all_pca[:len(jargon_embeddings)]
    layman_pca = all_pca[len(jargon_embeddings):len(jargon_embeddings)+len(layman_embeddings)]
    abstract_pca = all_pca[len(jargon_embeddings)+len(layman_embeddings):]

    tsne = TSNE(random_state=0)
    all_tsne = tsne.fit_transform(all_pca)
    
    # Split back into separate arrays
    jargon_tsne = all_tsne[:len(jargon_embeddings)]
    layman_tsne = all_tsne[len(jargon_embeddings):len(jargon_embeddings)+len(layman_embeddings)]
    abstract_tsne = all_tsne[len(jargon_embeddings)+len(layman_embeddings):]

    df_filtered["jargon_tsne_component_0"] = jargon_tsne[:, 0]
    df_filtered["jargon_tsne_component_1"] = jargon_tsne[:, 1]

    df_filtered["layman_tsne_component_0"] = layman_tsne[:, 0]
    df_filtered["layman_tsne_component_1"] = layman_tsne[:, 1]
    
    # Create a dataframe for abstracts
    abstract_df_to_plot = abstract_df_filtered.copy()
    abstract_df_to_plot["abstract_tsne_component_0"] = abstract_tsne[:, 0]
    abstract_df_to_plot["abstract_tsne_component_1"] = abstract_tsne[:, 1]

    df_to_plot = df_filtered
    
    # Ensure id column is the same type and categorical in both dataframes
    id_categories = sorted(df_to_plot["id"].unique())
    df_to_plot["id"] = pd.Categorical(df_to_plot["id"], categories=id_categories)
    abstract_df_to_plot["id"] = pd.Categorical(abstract_df_to_plot["id"], categories=id_categories)

    sns.set_theme(
        context="talk",
        style="white",
    )

    sns.scatterplot(
        data=df_to_plot,
        x="jargon_tsne_component_0",
        y="jargon_tsne_component_1",
        hue="id",
        legend=False,
        marker="o"
    )
    sns.scatterplot(
        data=df_to_plot,
        x="layman_tsne_component_0",
        y="layman_tsne_component_1",
        hue="id",
        legend=False,
        marker="^"
    )
    sns.scatterplot(
        data=abstract_df_to_plot,
        x="abstract_tsne_component_0",
        y="abstract_tsne_component_1",
        hue="id",
        legend=False,
        marker="D",
        linewidths=1,
        edgecolor="black"
    )
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks([])
    plt.yticks([])

    plt.title(args.model_name)

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight")
