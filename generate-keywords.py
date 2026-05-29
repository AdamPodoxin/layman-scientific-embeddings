import time
import json
import traceback
from typing import cast
from pathlib import Path
from datasets import Dataset, load_dataset
from vllm import LLM, SamplingParams


NUM_ABSTRACTS_TO_PROCESS = 2
KEYWORDS_PATH = Path("data") / "keywords"

MODEL = "unsloth/Qwen3-4B-Instruct-2507"
MAX_RESPONSE_TOKENS = 512
GPU_MEMORY_UTILIZATION = 0.85
MAX_MODEL_LEN = 8192

DATASET_PATH = "allenai/scirepeval"
DATASET_NAME = "scidocs_mag_mesh"

SYSTEM_PROMPT = \
"""
# Identity
You are a Senior Science Communicator and Semantic Engineer. 
Your expertise is in "Deep Simplification", translating high-level scientific jargon into layman's concepts without losing the underlying causal logic or specificity.

# Task
Analyze the provided scientific abstract and extract 15 distinct concept pairs. 
- A "Jargon" term must be the precise technical term used in the field.
- A "Layman" term must be a simplified, non-expert conceptual equivalent.

# Categorization Strategy
You must provide 5 pairs for each of the following categories:
1. CORE ENTITIES (The "What"): Primary biological, physical, or theoretical objects.
2. METHODOLOGIES/PROCESSES (The "How"): Experimental techniques or natural mechanisms.
3. OUTCOMES/METRICS (The "Result"): Statistical findings, specific observations, or consequences.

# Constraints
- Output ONLY a valid JSON object. 
- DO NOT include conversational filler, markdown formatting (no ```json blocks), or "thinking" tags.
- Keep layman terms concise, small phrases, not full sentences.
- If a jargon term is easy to understand for a layman (e.g. made from normal words) or if it describing a type of entity (e.g. type of animal), then keep the layman term identical to the jargon term.
- Ensure 1:1 conceptual mapping.
- If the abstract is too short for 15 pairs, focus on the most complex terms rather than generating filler.
"""


def create_user_prompt(title: str, abstract: str):
    return \
    f"""
    ### INPUT TITLE AND ABSTRACT
    {title}
    {abstract}

    ### INSTRUCTIONS
    Generate 15 Jargon-to-Layman keyword pairs from the title and abstract above. Follow the Core Entities, Methodologies, and Outcomes taxonomy.

    ### EXAMPLE FORMATTING (FEW-SHOT)
    {{
    "core_entities": [
        {{"jargon": "Murine models", "layman": "Laboratory mice for testing"}},
        {{"jargon": "Cytokine storm", "layman": "Overactive and dangerous immune response"}}
        {{"jargon": "Foam nests", "layman": "Foam nests"}}
        {{"jargon": "Mourning cuttlefish", "layman": "Mourning cuttlefish"}}
    ],
    "methodologies": [
        {{"jargon": "CRISPR-Cas9", "layman": "Gene editing tool"}},
        {{"jargon": "Double-blind study", "layman": "Test where neither the doctor nor patient knows the treatment"}}
    ],
    "outcomes": [
        {{"jargon": "P-value < 0.05", "layman": "Significant result"}},
        {{"jargon": "Apoptosis", "layman": "Programmed cell death"}}
    ]
    }}
    """


def create_messages_for_pipe(row: dict):
    title = row["title"]
    abstract = row["abstract"]

    if not title or not abstract:
        return {"prompt": None}

    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": create_user_prompt(title, abstract)},
    ]

    return {"prompt": prompt}


def get_keywords_from_response(assistant_response: str) -> dict:
    return json.loads(assistant_response)


def main():
    KEYWORDS_PATH.mkdir(parents=True, exist_ok=True)
    
    ds: Dataset = cast(
        Dataset,
        load_dataset(DATASET_PATH, DATASET_NAME, split="evaluation"),
    )

    ds = ds.filter(lambda row: row["title"] is not None and row["abstract"] is not None)
    ds = ds.take(NUM_ABSTRACTS_TO_PROCESS)

    # Don't re-generate keywords for already done abstracts 
    ds_to_generate = ds.filter(lambda row: not (KEYWORDS_PATH / f"{row["doc_id"]}.json").exists())
    ds_to_generate = ds_to_generate.map(create_messages_for_pipe)

    num_abstracts = ds_to_generate.shape[0]

    print("Starting keyword generation for", num_abstracts, "abstracts")

    if num_abstracts == 0:
        print("No abstracts need keyword generation; all selected abstracts are already processed.")
        return

    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
    )

    start_time = time.time()

    doc_ids = ds_to_generate["doc_id"]
    messages = ds_to_generate["prompt"]
    responses = llm.chat(
        messages=messages,
        sampling_params=SamplingParams(
            max_tokens=MAX_RESPONSE_TOKENS,
        ),
        use_tqdm=True,
    )

    for doc_id, response in zip(doc_ids, responses):
        filename = f"{doc_id}.json"
        keywords_file_path = KEYWORDS_PATH / filename

        try:
            keywords = get_keywords_from_response(response.outputs[0].text)
            
            with open(keywords_file_path, "w+") as file:
                file.write(json.dumps(keywords, indent=2))
            
            print("Created keywords for", filename)
        except:
            print("Error for abstract", doc_id)
            traceback.print_exc()

    print("Generated keywords for", num_abstracts, "abstracts in", time.time() - start_time, "seconds")


if __name__ == "__main__":
    main()
