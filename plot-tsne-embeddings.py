import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_finetuned_qwen


def read_keywords_file(path: Path) -> dict:
    with open(path) as f:
        return json.loads(f.read())


def get_keywords_from_document(document: dict):
    keyword_pairs: list[dict[str, str]] = list(document["core_entities"]) \
                    + list(document["methodologies"]) \
                    + list(document["outcomes"])
    
    return keyword_pairs


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
        "--input",
        type=Path,
        default=Path("data/test_keywords"),
        help="Path to keywords directory",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("plots/tsne/vanilla-scibert.png"),
        help="Output image path",
    )
    p.add_argument(
        "--dataset-path",
        type=str,
        default="allenai/scirepeval",
        help="HF dataset path",
    )
    p.add_argument(
        "--dataset-name",
        type=str,
        default="scidocs_mag_mesh",
        help="HF dataset name",
    )
    p.add_argument(
        "--dataset-split",
        type=str,
        default="evaluation",
        help="HF dataset split",
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

    df = pd.DataFrame(data={ "path": [path for path in Path(args.input).iterdir()] })
    df["id"] = df["path"].apply(lambda path: path.stem)
    df["document"] = df["path"].apply(read_keywords_file)
    df["keywords"] = df["document"].apply(get_keywords_from_document)
    df["jargon"] = df["keywords"].apply(lambda keyword_pairs: list(pair["jargon"] for pair in keyword_pairs))
    df["layman"] = df["keywords"].apply(lambda keyword_pairs: list(pair["layman"] for pair in keyword_pairs))

    ds = load_dataset(
        path=str(args.dataset_path),
        name=str(args.dataset_name),
        split=str(args.dataset_split)
    )
    ds_df = ds.to_pandas()

    df = pd.merge(left=df, right=ds_df, left_on="id", right_on="doc_id")

    # Filter to documents we want to plot BEFORE encoding
    ids = list(set(df["id"]))
    ids_to_plot = ids[:int(args.num_docs)]
    
    df_filtered = df[df["id"].isin(ids_to_plot)]
    ds_df_filtered = ds_df[ds_df["doc_id"].isin(ids_to_plot)]

    df_filtered = df_filtered.explode(["jargon", "layman"])

    jargon_embeddings = model.encode(df_filtered["jargon"].to_list())
    layman_embeddings = model.encode(df_filtered["layman"].to_list())
    
    # Encode abstracts for filtered documents only
    abstract_embeddings = model.encode(ds_df_filtered["abstract"].to_list())

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
    abstract_df_to_plot = ds_df_filtered.copy()
    abstract_df_to_plot["id"] = abstract_df_to_plot["doc_id"]
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
