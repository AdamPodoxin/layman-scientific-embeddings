# LaySciSearch: Bridging Jargon and Layman's Keywords in Scientific Embeddings

A research project that fine-tunes embedding models to understand and bridge the gap between scientific jargon and layman's terminology, enabling more accessible search and discovery of scientific literature.

## Overview

This project extracts jargon-layman concept pairs from scientific abstracts and uses them to fine-tune embedding models (Qwen and SciBERT). The fine-tuned models are evaluated on their ability to retrieve semantically related abstracts using different search strategies (keyword, title, and abstract-based).

### Key Objectives

- Extract scientific jargon and their layman's equivalents from abstracts using LLMs
- Fine-tune embedding models on jargon-layman pairs to improve semantic understanding
- Compare model performance using keyword, title, and abstract search strategies
- Generate and visualize embeddings to understand semantic relationships

## Project Structure

```
.
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── utils.py                           # Utility functions (model loading, etc.)
│
├── Data Generation Pipeline:
│   ├── generate-keywords.py           # Extract jargon-layman pairs from abstracts
│   ├── create-pairs.py                # Create training pairs from extracted keywords
│   └── export-test-keywords.py        # Prepare test data
│
├── Model Fine-tuning:
│   ├── finetune-vanilla-qwen.py       # Vanilla Qwen model fine-tuning
│   ├── finetune-vanilla-scibert.py    # Vanilla SciBERT model fine-tuning
│   ├── finetune-jargon-layman-qwen.py # Qwen fine-tuning on jargon-layman pairs
│   └── finetune-jargon-layman-scibert.py # SciBERT fine-tuning on jargon-layman pairs
│
├── Evaluation:
│   ├── keyword_search_evaluation.py   # Evaluate using keyword-based search
│   ├── title_search_evaluation.py     # Evaluate using title-based search
│   ├── abstract_search_evaluation.py  # Evaluate using abstract-based search
│   └── test-model-similarities.py     # Test embedding similarities
│
├── Analysis & Visualization:
│   ├── plot-tsne-embeddings.py        # Generate t-SNE plots of embeddings
│   ├── test.ipynb                     # Testing and experimentation
│   ├── keyword_search_evaluation.ipynb
│   ├── title_search_evaluation.ipynb
│   └── abstract_search_evaluation.ipynb
│
└── Data and Models:
    ├── data/
    │   ├── keywords/              # Extracted jargon-layman concept pairs
    │   ├── pairs/                 # Generated training pairs
    │   │   ├── abstract-jargon/
    │   │   ├── abstract-layman/
    │   │   ├── title-jargon/
    │   │   ├── title-layman/
    │   │   ├── jargon-jargon/
    │   │   ├── layman-layman/
    │   │   └── jargon-layman/    # Main training data
    │   └── test_keywords/         # Test set keywords
    │
    ├── models/
    │   ├── vanilla-qwen/          # Base Qwen model fine-tuned
    │   ├── jargon-layman-qwen/    # Qwen fine-tuned on jargon-layman
    │   ├── vanilla-scibert/       # Base SciBERT fine-tuned
    │   └── jargon-layman-scibert-*.*/  # SciBERT variants
    │
    └── plots/
        └── tsne/                  # t-SNE visualization plots
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA (for GPU acceleration, recommended)

### Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd layman-scientific-embeddings
```

2. Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Pipeline

### 1. Generate Keywords from Abstracts

Extract jargon-layman concept pairs from scientific abstracts using an LLM:

```bash
python generate-keywords.py
```

This script:

- Uses Qwen3-4B-Instruct to analyze scientific abstracts
- Extracts 15 concept pairs per abstract (jargon + layman equivalent)
- Categorizes pairs into: core entities, methodologies, and outcomes
- Saves results to `data/keywords/`

### 2. Create Training Pairs

Generate training data with various pair combinations:

```bash
python create-pairs.py
```

Creates:

- `jargon-layman` pairs (main training data)
- Abstract/title-based jargon/layman pairs
- Same-type pairs (jargon-jargon, layman-layman)

### 3. Export Test Keywords

Prepare test data for evaluation:

```bash
python export-test-keywords.py
```

## Model Fine-tuning

### Fine-tune on All Data

Start with vanilla model training:

```bash
python finetune-vanilla-qwen.py
python finetune-vanilla-scibert.py
```

### Fine-tune Further on Jargon-Layman Pairs

Train models with jargon-layman terminology mapping:

```bash
python finetune-jargon-layman-qwen.py
python finetune-jargon-layman-scibert.py
```

Fine-tuning details:

- **Models**: Qwen3-Embedding-0.6B, SciBERT
- **Technique**: LoRA (Low-Rank Adaptation) for efficient fine-tuning
- **Quantization**: 4-bit for memory efficiency
- **Training**: Contrastive learning on concept pairs

## Evaluation

### Keyword-based Search Evaluation

Evaluate model performance using extracted keywords:

```bash
python keyword_search_evaluation.py --model-path models/jargon-layman-qwen/
```

```bash
jupyter notebook keyword_search_evaluation.ipynb
```

### Title-based Search Evaluation

Evaluate using document titles:

```bash
python title_search_evaluation.py --model-path models/jargon-layman-qwen/
```

```bash
jupyter notebook title_search_evaluation.ipynb
```

### Abstract-based Search Evaluation

Evaluate using full abstract text:

```bash
python abstract_search_evaluation.py --model-path models/jargon-layman-qwen/
```

```bash
jupyter notebook abstract_search_evaluation.ipynb
```

### Model Similarity Testing

Test embedding similarities and relationships:

```bash
python test-model-similarities.py
```

## Visualization

### Generate t-SNE Plots

Create 2D t-SNE visualizations of embeddings:

```bash
python plot-tsne-embeddings.py \
  --model-path models/jargon-layman-qwen/ \
  --model-name LaySciSearch-jargon-layman-qwen \
  --output plots/tsne/jargon-layman-qwen.png
```

Options:

- `--model-path`: Path to model directory
- `--model-name`: Display name for plot
- `--output`: Output image path
- `--sample-size`: Number of embeddings to visualize (default: 1000)

## Key Components

### Utility Functions (`utils.py`)

- `load_finetuned_qwen(adapter)`: Load Qwen model with LoRA adapters
  - Supports 4-bit quantization for memory efficiency
  - Loads pre-trained adapters from model path

### Data Structure

Each keyword file contains:

```json
{
  "document_id": "...",
  "core_entities": [
    {"jargon": "term1", "layman": "simple_term1"},
    ...
  ],
  "methodologies": [...],
  "outcomes": [...]
}
```

### Training Pairs

Training pairs are generated as:

- Positive pairs: (jargon, layman) - semantically related
- Negatives: All other pairs from different abstracts

## Models Used

### Qwen3-Embedding-0.6B

- Lightweight embedding model (600M parameters)
- Efficient for deployment
- Fine-tuned with LoRA adapters

### SciBERT

- Domain-specific BERT for scientific text
- Pre-trained on scientific papers
- Multiple fine-tuned variants available

## Dataset

- **Source**: Allen AI SciRepEval (SCIDOCS)
- **Domain**: Scientific abstracts from MAG/MESH
- **Size**: ~1,000 abstracts processed
- **Pairs**: 15 jargon-layman pairs per abstract

## Requirements

Key dependencies:

- `torch`: Deep learning framework
- `sentence-transformers`: Embedding models
- `transformers`: LLM for keyword generation
- `datasets`: Data loading and processing
- `peft`: LoRA for efficient fine-tuning
- `safetensors`: Safe tensor serialization
- `matplotlib`: Visualization
- `scikit-learn`: t-SNE and utilities
- `pandas`: Data manipulation

See `requirements.txt` for complete list.

## Usage Examples

### Load a Fine-tuned Model

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("models/jargon-layman-qwen")
embeddings = model.encode(["your text here"])
```

### Search for Similar Abstracts

```python
from sentence_transformers.util import semantic_search

query_embedding = model.encode("your query")
corpus_embeddings = model.encode(corpus)
results = semantic_search(query_embedding, corpus_embeddings, top_k=5)
```

### Extract and Visualize Embeddings

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

embeddings = model.encode(texts)
tsne = TSNE(n_components=2)
reduced = tsne.fit_transform(embeddings)
plt.scatter(reduced[:, 0], reduced[:, 1])
plt.show()
```

## Configuration

### Fine-tuning Parameters

Edit the respective `finetune-*.py` files to adjust:

- `MINI_BATCH_SIZE`: Batch size for training
- Learning rate and warmup steps
- Number of epochs
- LoRA parameters (rank, alpha)

### Evaluation Parameters

Edit evaluation scripts to modify:

- `TOP_K_ABSTRACTS`: Number of results to retrieve
- `NUM_KEYWORDS_PER_ABSTRACT`: Keywords per document
- `SAMPLE_PROPORTION`: Proportion of pairs to use

## Results

Models are evaluated on:

TODO fix

- **Mean Reciprocal Rank (MRR)**: Quality of top results
- **Recall@K**: Percentage of relevant docs in top-K
- **Semantic similarity**: Cosine similarity between concept pairs

Performance varies by:

- Search strategy (keyword > title > abstract)
- Model (Qwen vs SciBERT)
- Fine-tuning approach (vanilla vs jargon-layman)

## Future Work

TODO: fix

- Expand to more scientific domains
- Multi-lingual jargon-layman mappings
- Interactive web interface for semantic search
- Comparison with other domain-specific embeddings
- Fine-grained evaluation metrics

## References

- [SciRepEval Dataset](https://github.com/allenai/scirepeval)
- [Sentence Transformers](https://www.sbert.net/)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [t-SNE](https://scikit-learn.org/stable/modules/manifold.html#t-sne)
