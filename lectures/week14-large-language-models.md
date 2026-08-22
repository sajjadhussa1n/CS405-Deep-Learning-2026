# Week 14 — Large Language Models (LLMs)

*Companion notes for [`slides/lecture_week14.pdf`](../slides/lecture_week14.pdf)*

## Why this week matters

Week 13 gave us the Transformer architecture. This week is about what happens when you take that architecture and scale it up dramatically — in parameter count, training data, and compute — and what practical ecosystem has grown up around the resulting models. This is also the first week to move beyond pure architecture and squarely into practical deployment concerns: how these models are trained and adapted, what they're good and bad at, and the ethical and safety issues that come with using them.

## 1. What makes a language model "large"?

A **Large Language Model (LLM)** is a neural network — built from the Transformer components of Week 13 — trained on massive amounts of text data to understand and generate human-like language. The "large" in LLM refers to three things scaling together simultaneously: **model size** (billions to trillions of parameters — GPT-3 has 175 billion, GPT-4 is estimated around 1.7 trillion), **training data** (hundreds of billions to trillions of tokens, drawn from sources like the internet, books, and code, often multilingual), and **compute** (thousands of GPUs/TPUs running for weeks to months, at a cost of millions of dollars). The slides summarize this succinctly: `LLMs = Scale + Data + Compute → Emergent Abilities` — meaning that as these three factors grow together, models don't just get incrementally better at what smaller models already did; qualitatively new capabilities (few-shot learning, chain-of-thought reasoning, and so on) tend to appear that weren't reliably present at smaller scale.

## 2. Three architectural families

Building on the Transformer's encoder and decoder components from Week 13, LLMs fall into three broad architectural families, each suited to different kinds of tasks:

| Type | Architecture | Examples | Best for |
|---|---|---|---|
| Encoder-only | bidirectional | BERT, RoBERTa, ALBERT | understanding, classification |
| Decoder-only | autoregressive | GPT-3/4, LLaMA, Mistral | generation, chat, code |
| Encoder-decoder | full seq2seq | T5, BART, Pegasus | translation, summarization |

The intuitive distinction: **encoder-only** models ("read and understand") use the Transformer encoder's bidirectional self-attention from Week 13 to build a deep contextual representation of an *entire* input, well suited to tasks like classification or extracting information, but not naturally suited to generating new text word by word. **Decoder-only** models ("read and continue") use only the causally-masked, autoregressive half of the architecture, and are trained purely to predict the next token given everything before it — this is the dominant architecture for today's chat-style and code-generation models. **Encoder-decoder** models ("read and transform") use the full architecture from Week 13, with a separate encoder processing the input and a decoder generating a transformed output via cross-attention, naturally suited to tasks like translation and summarization where the output is a genuinely different sequence built *from* the input rather than a continuation of it. The right architectural family depends on the task at hand.

### Categorizing by access and by size

Beyond architecture, LLMs are also commonly categorized by **access model** — open source (weights are publicly released and can be downloaded and modified, e.g., LLaMA, Mistral, Falcon), open access (usable only via an API, with the underlying weights kept private, e.g., GPT-4, Claude, Gemini), and research access (available only through a limited application process, e.g., early releases like PaLM) — and by **parameter count**: small (under 1B, e.g., DistilBERT), medium (1B–10B, e.g., GPT-2 medium), large (10B–100B, e.g., LLaMA 2 70B), and very large (100B–1T+, e.g., GPT-4). An important caveat worth internalizing: **bigger doesn't always mean better for every task** — a much smaller, task-specific fine-tuned model can outperform a giant general-purpose model on a narrow task, while being far cheaper to run.

## 3. How LLMs are trained: pre-training then fine-tuning

### Pre-training

**Pre-training** is the initial, expensive phase in which a model learns general language ability from massive amounts of *unlabeled* text, typically by predicting masked or missing tokens (encoder-style, e.g., BERT's **masked language modeling** — given "The [MASK] sat on the [MASK]," predict "cat" and "mat," with a cross-entropy loss computed only at the masked positions) or by predicting the next token given everything before it (decoder-style **causal/autoregressive language modeling**, as used by the GPT family). Through this single, simple objective — repeated across an enormous amount of text — the model implicitly learns vocabulary and syntax, grammar and word order, a great deal of world knowledge and facts, useful reasoning patterns, and cultural context, all without ever being explicitly told any of these things. Pre-training is, in effect, learning language (and a great deal about the world described by that language) essentially from scratch.

### Fine-tuning

**Fine-tuning** is further training of an already pre-trained model on smaller, task-specific *labeled* data, adapting its general capability to a specific need. Several distinct fine-tuning strategies exist:

| Type | Description | Typical data size |
|---|---|---|
| Full fine-tuning | update all parameters | 10K–100K examples |
| Instruction fine-tuning | learn to follow instructions | 10K–1M (instruction, response) pairs |
| PEFT (e.g., LoRA) | train small adapters only | 100–10K examples |
| RLHF | learn from human preference comparisons | 10K–100K comparisons |

**Parameter-Efficient Fine-Tuning (PEFT)** deserves special attention because it's become the practical default for adapting large models: rather than updating every one of a model's (potentially hundreds of billions of) parameters, methods like **LoRA (Low-Rank Adaptation)** insert small, trainable low-rank matrices into specific layers (commonly the attention projection layers) while keeping the vast majority of the original model **frozen**. This can train as little as 0.1%–1% of the total parameter count, drastically reducing memory usage and training time while still achieving strong task-specific adaptation — a technique we'll see used directly in the code example in Section 6.

## 4. Zero-shot and few-shot learning

One of the most practically important emergent capabilities of large pre-trained models is the ability to perform new tasks *without any gradient-based fine-tuning at all*, simply by how the prompt is phrased. In **zero-shot learning**, the model performs a task given only a natural-language instruction and no examples — e.g., prompting "Classify as positive or negative: 'Great product!'" and getting "positive" back, purely from the model's pre-trained understanding of the instruction and the text. In **few-shot learning** (also called **in-context learning**), a handful of example input-output pairs are included directly in the prompt itself — e.g., showing "Great!" → Positive and "Terrible!" → Negative before asking about "Average" — and the model infers the pattern and applies it to the new input, again with **no weight updates whatsoever**. This matters enormously in practice: it lets a single pre-trained model be repurposed for a huge range of tasks on the fly, leveraging everything it absorbed during pre-training, without the cost or delay of fine-tuning a separate model for every new task.

## 5. Retrieval-Augmented Generation (RAG)

Even a very capable LLM has two structural limitations: its knowledge is frozen at whatever point its training data was collected, and it has no built-in way to cite where an answer came from. **Retrieval-Augmented Generation (RAG)** addresses both by combining the LLM with an external knowledge source at inference time, following a three-step pipeline: **retrieve** relevant documents from an external source (typically a vector database of document embeddings, searched for semantic similarity to the query); **augment** the prompt by inserting the retrieved text as additional context; and **generate** the final answer, now conditioned on that retrieved, up-to-date context rather than relying purely on what the model memorized during pre-training. For example, asked "What did the CEO say in yesterday's meeting?", a RAG system retrieves the actual meeting transcript, adds it to the prompt, and generates an answer grounded in — and potentially quoting directly from — that retrieved text. RAG offers several concrete benefits over relying on a model's frozen pre-trained knowledge alone: access to up-to-date information without any retraining, reduced hallucination (since answers are grounded in retrieved text rather than purely generated from parametric memory), the ability to provide citations for verification, and secure access to private or internal documents that were never part of the model's training data.

## 6. Challenges and risks

### Hallucination

**Hallucination** is when a model generates fluent, confident-sounding text that is factually wrong — for example, confidently stating that Einstein won the 1875 Nobel Prize in Physics, when in fact no Nobel Prizes were awarded that year (the prizes weren't established until 1901), and Einstein actually won in 1921. This happens because LLMs are fundamentally **pattern matchers over their training data, not verified knowledge bases** — they have no built-in mechanism for distinguishing true statements from false but plausible-sounding ones, their training data itself often contains contradictions, and the training objective rewards fluent, plausible-sounding continuations rather than directly rewarding factual accuracy. Mitigations include RAG (grounding answers in retrieved, verifiable text), self-consistency checks (generating multiple answers and checking agreement), requiring citation generation, and reinforcement learning objectives that specifically reward factuality.

### Bias and fairness

**Bias** in LLMs refers to systematic, unfair patterns in model outputs that disadvantage particular groups — for example, gender bias (associating "CEO" with "he" and "nurse" with "she"), racial/ethnic bias (certain names being associated with crime or poverty in generated text), cultural bias (a Western-centric default worldview, e.g., assuming Christmas rather than other cultural or religious holidays), socioeconomic bias (penalizing non-standard English), and political bias (systematically leaning toward particular ideological positions). It's useful to keep **bias** (the measurable, empirical problem — a pattern that can actually be detected in outputs) conceptually distinct from **fairness** (the normative, value-based *goal* of what an equitable outcome should look like) — fixing the former doesn't automatically resolve disagreements about the latter.

### Safety, toxicity, and privacy

Models can generate harmful, offensive, or dangerous content — including hate speech and harassment, instructions facilitating illegal activity, or content promoting self-harm — mitigated primarily through safety-focused fine-tuning (notably RLHF, reinforcement learning from human feedback), content filtering, and usage policies. Separately, **privacy** concerns arise because models can sometimes **memorize** and later reproduce specific pieces of their training data (potentially including personally identifiable information), users can inadvertently leak sensitive information by pasting it into a prompt ("prompt leakage"), and adversarial "model inversion" attacks can sometimes be used to extract specific training examples from a model. Mitigations include differential privacy techniques during training, deduplication of training data, and automated redaction of personally identifiable information.

### Computational cost and other limitations

Training frontier LLMs is extraordinarily expensive — estimated at $4–12 million for GPT-3 (175B parameters), $2–5 million for LLaMA 2 70B, and $100–200 million for GPT-4 (roughly 1.7T parameters) — with a real environmental cost as well, since training runs of this scale can emit hundreds of tons of CO2. Beyond cost, LLMs face several other structural limitations worth naming: **catastrophic forgetting** (fine-tuning a model on a new task can degrade or erase capabilities it previously had), a **lack of true understanding** (LLMs perform sophisticated statistical pattern matching over text, which is a fundamentally different thing from genuine reasoning or comprehension, however convincing the output may appear), and a **limited context window** (a hard cap on how much text a model can process or "remember" within a single interaction, which constrains its ability to work with very long documents).

## 7. The open ecosystem: Hugging Face

**Hugging Face** is often described as "the GitHub for machine learning" — a central platform for sharing pre-trained models, datasets, and interactive applications. Its core components include the **Model Hub** (a repository of 500,000+ pre-trained models), **Datasets** (100,000+ shared datasets), **Spaces** (hosting for interactive ML demo applications), the **Transformers** library (a unified Python interface for loading and using models across both PyTorch and TensorFlow), **PEFT** (a library implementing parameter-efficient fine-tuning methods like LoRA and AdaLoRA), and **Accelerate** (tooling for distributed, multi-GPU training). Popular open models available through the Hub include the LLaMA 2/3 family (Meta), Mistral and Mixtral, Gemma (Google), and models like Zephyr, Phi-2, Falcon, and BLOOM.

### Using Hugging Face in practice

The library is designed to make using a pre-trained model require very little code. A high-level, task-oriented `pipeline` can run a common task in just a couple of lines:

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I love this product!")
# Output: [{'label': 'POSITIVE', 'score': 0.999}]
```

More generally, any model on the Hub can be loaded directly by name using the appropriate `Auto*` classes:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")
```

And, connecting directly back to the PEFT/LoRA discussion in Section 3, fine-tuning with LoRA on top of a loaded base model is similarly concise:

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8, lora_alpha=32,
    target_modules=["q_proj", "v_proj"]  # apply LoRA to the attention Q/V projections
)
peft_model = get_peft_model(base_model, lora_config)
# Only trains approximately 0.1% of parameters!
```

Note that `target_modules=["q_proj", "v_proj"]` inserts the small trainable low-rank adapter matrices specifically into the query and value projection matrices of the attention mechanism from Week 13 — LoRA doesn't touch the model's other weights at all. The practical benefits of this approach are substantial: drastically reduced memory usage during fine-tuning (since gradients and optimizer state only need to be tracked for the small adapter matrices, not the full model), faster training, the ability to easily swap in different lightweight adapters for different downstream tasks without duplicating the entire base model, and — because the adapter matrices can be merged back into the original weights after training — no added latency at inference time.

## 8. Model licenses matter

Not all "open" models can be used the same way — the license attached to a model's weights determines what you're actually allowed to do with it, and this varies significantly:

| License | Commercial use? | Examples |
|---|---|---|
| MIT / Apache 2.0 | yes | BERT, T5, GPT-2, Mistral |
| LLaMA 2/3 license | yes, with restrictions | LLaMA 2, LLaMA 3 |
| LLaMA 1 license | no (research only) | LLaMA 1 |
| OpenAI (weights not released) | no | GPT-3, GPT-4 |

Before using any model in a project — and especially before deploying anything commercially — check the license listed on the model's card, verify commercial usage rights explicitly, review any attribution requirements, and check for usage-scale restrictions (for example, Meta's LLaMA licenses have historically required separate approval for products with more than 700 million monthly active users). **Always check the license before deploying.**

## Key takeaways

Large Language Models are Transformer-based (Week 13) networks scaled up dramatically in parameters, training data, and compute, falling into three architectural families — encoder-only (understanding tasks), decoder-only (generation tasks), and encoder-decoder (transformation tasks) — chosen based on what the task actually requires. They're built through a two-phase training process: broad, self-supervised **pre-training** on massive unlabeled text, followed by targeted **fine-tuning** (full, instruction-based, parameter-efficient via LoRA, or preference-based via RLHF) to specialize the model for particular tasks or behaviors, with zero-shot and few-shot prompting offering a third, gradient-free way to adapt a pre-trained model on the fly. Retrieval-Augmented Generation extends a model's effective knowledge and grounds its answers in retrieved, verifiable text. Despite their power, LLMs carry serious, well-documented risks — hallucination, bias, safety/toxicity, privacy leakage, high computational and environmental cost, catastrophic forgetting, and a hard limit on context length — that any responsible deployment needs to actively account for, not treat as an afterthought. The Hugging Face ecosystem has become the de facto standard for accessing, running, and fine-tuning open models in practice, but using any specific model responsibly also means checking its license before you build on it.
