import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


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
        default="models/vanilla-finetuned",
        help="The HF model id or local directory",
    )
    p.add_argument(
        "--model-name",
        type=str,
        default="LaySciSearch-vanilla-SciBERT",
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
        default=Path("plots/tsne/vanilla-tsne.png"),
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
        "--num-docs-plot",
        type=int,
        default=8,
        help="Number of documents whose keywords to plot",
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model = SentenceTransformer(args.model_path)

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

    df = df.explode(["jargon", "layman"])

    jargon_embeddings = model.encode(df["jargon"].to_list())
    layman_embeddings = model.encode(df["layman"].to_list())

    pca = PCA()
    jargon_pca = pca.fit_transform(jargon_embeddings)
    layman_pca = pca.fit_transform(layman_embeddings)

    tsne = TSNE(random_state=0)
    jargon_tsne = tsne.fit_transform(jargon_pca)
    layman_tsne = tsne.fit_transform(layman_pca)

    df["jargon_tsne_component_0"] = jargon_tsne[:, 0]
    df["jargon_tsne_component_1"] = jargon_tsne[:, 1]

    df["layman_tsne_component_0"] = layman_tsne[:, 0]
    df["layman_tsne_component_1"] = layman_tsne[:, 1]

    ids = list(set(df["id"]))
    ids_to_plot = ids[:int(args.num_docs_plot)]

    df_to_plot = df[df["id"].isin(ids_to_plot)]

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
        marker="D"
    )
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks([])
    plt.yticks([])

    plt.title(args.model_name)

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight")
