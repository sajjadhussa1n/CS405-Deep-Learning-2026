# Lab 13 — LLMs with Hugging Face

**Matches:** [Week 14 — Large Language Models](../lectures/week14-large-language-models.md)
**Goal:** Get hands-on with the Hugging Face ecosystem — pipelines, zero/few-shot prompting, LoRA fine-tuning, and a minimal RAG pipeline.

## Setup

```bash
pip install transformers datasets peft accelerate torch
```

## Step 1 — The `pipeline` API

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
print(classifier("I love this course!"))

generator = pipeline("text-generation", model="distilgpt2")
print(generator("The best way to learn deep learning is", max_new_tokens=30))
```

Try at least 3 different pipeline tasks (e.g., `"summarization"`, `"question-answering"`, `"zero-shot-classification"`) on inputs of your choosing and record the outputs.

## Step 2 — Zero-shot and few-shot prompting

Using the `"text-generation"` pipeline with a small model (e.g., `distilgpt2` or `gpt2`), construct: (a) a **zero-shot** prompt (instruction only, no examples) for a simple classification task, and (b) a **few-shot** prompt (2–3 worked examples followed by a new input) for the same task. Compare the outputs and report whether few-shot prompting improved the model's answer, as the lecture notes suggest it should.

## Step 3 — Load and inspect a model directly

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

inputs = tokenizer("This lab is great!", return_tensors="pt")
outputs = model(**inputs)
print(outputs.logits.softmax(dim=-1))
```

Report the total parameter count of this model (`sum(p.numel() for p in model.parameters())`) and where it falls in the "small/medium/large" size categories from the lecture.

## Step 4 — Fine-tune with LoRA

Pick a small classification dataset from the `datasets` library (e.g., `imdb`, subsampled to a few hundred examples for speed) and fine-tune a small base model (e.g., `distilbert-base-uncased`) using LoRA:

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForSequenceClassification

base_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8, lora_alpha=16,
    target_modules=["q_lin", "v_lin"],  # DistilBERT's attention projection names
)
peft_model = get_peft_model(base_model, lora_config)
peft_model.print_trainable_parameters()
```

Train it (e.g., with `transformers.Trainer` or a plain PyTorch loop) for a few epochs on your small dataset, and report accuracy before and after fine-tuning, along with the exact percentage of trainable parameters LoRA reports.

## Step 5 — A minimal RAG pipeline

1. Assemble a small "knowledge base" of 5–10 short text passages (e.g., a handful of paragraphs from this course's own `lectures/` folder — that's a convenient, ready-made corpus!).
2. Embed each passage (you can reuse a `sentence-transformers` model such as `all-MiniLM-L6-v2`, or your own Week 11-style embeddings for a rougher approximation) and store the embeddings.
3. Given a query, embed it the same way, compute cosine similarity against every passage, and retrieve the top 1–2 most similar passages.
4. Construct a prompt that includes the retrieved passage(s) as context, followed by the question, and feed it to your Step 1/3 generation model.

```python
from sentence_transformers import SentenceTransformer, util

embedder = SentenceTransformer("all-MiniLM-L6-v2")
passages = [...]  # your knowledge base
passage_embeddings = embedder.encode(passages, convert_to_tensor=True)

def retrieve(query, k=1):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, passage_embeddings)[0]
    top_k = scores.topk(k).indices.tolist()
    return [passages[i] for i in top_k]
```

Run at least 5 example queries through your full pipeline (retrieve → augment prompt → generate), and include one example where the retrieved context is clearly what allows the model to answer correctly (versus asking the same question directly with no retrieved context).

## Checkpoint questions

1. In Step 4, what percentage of the base model's parameters did LoRA actually train? How does that compare to the "0.1% to 1%" figure quoted in the lecture notes?
2. In Step 5, did retrieval genuinely change or improve the generated answer for your chosen examples? If your knowledge base is small, is it possible the base model already "knew" the answer without retrieval — how would you design a better test of RAG's value?
3. Try one of your Step 3 pipeline outputs where the model produces a plausible-sounding but incorrect answer. Is this an example of hallucination as described in the lecture? What mitigation from the lecture notes (RAG, self-consistency, citation) would most directly address it?
