🧠 Semantic Memory Dataset Pipeline – Project Overview

This project builds a large-scale semantic triplet dataset and trains a vector-based memory system using LoRA fine-tuning. It uses Meta LLaMA 3 models hosted via llama.cpp and produces realistic, domain-labeled training samples in three stages:
1. anchor-builder.py – Anchor Generation

🔍 Purpose:
Generates thousands of semantically realistic “anchor” thoughts for each memory domain. These represent user-like internal thoughts, intentions, or observations.

🧱 Scope:

    Domains: reflection, goals, task, observation, tool, knowledge, chat

    Output: ~1,000–1,200 anchors per domain

    Prompting: Uses few-shot examples to generate 20 domain-matching thoughts at a time

⚙️ Technologies:

    Model: Meta-LLaMA-3-8B-Instruct.Q6_K.gguf via llama-server (http://localhost:8080/v1/chat/completions)

    Libraries: requests, matplotlib, argparse, json, multiprocessing

📤 Output:
Per-domain .jsonl files, e.g.

{"domain": "goals", "text": "I want to improve my public speaking skills."}

2. vector-bank-generator.py – Triplet Generation

🔍 Purpose:
Transforms anchors into contrastive triplets for vector embedding training:

    Anchor

    Positive (semantically similar)

    Negative (semantically distant)

🧱 Scope:

    Uses 6 anchors at a time in each generation batch

    Generates ~250,000 total triplets

    Computes token lengths for analysis

⚙️ Technologies:

    Model: Meta-LLaMA-3-8B-Instruct.Q6_K.gguf via load-balanced llama-server ports (8080–8083)

    Tokenizer: Hugging Face AutoTokenizer

    Libraries: threading, ThreadPoolExecutor, json, argparse, requests, matplotlib

📤 Output:
Unified triplet dataset .jsonl, e.g.

{
  "anchor": "I want to improve my focus during meetings.",
  "positive": "I'm trying to stay more present when others are talking.",
  "negative": "The grocery store was packed this morning.",
  "domain": "goals",
  "token_length": 72
}

3. vector-lora-trainer.py – LoRA Fine-Tuning

🧪 Purpose:
Trains a semantic embedding model using the contrastive triplets dataset via LoRA fine-tuning on Mistral 7B.

🧱 Scope:

    Trains using anchor+positive pairs

    Ignores negatives during loss computation

    Supports checkpoint resumption and autosaving

    Runs on multi-GPU (DDP) with mixed precision and gradient accumulation

⚙️ Technologies:

    Model: mistralai/Mistral-7B-Instruct-v0.1

    LoRA: peft.LoraConfig

    Libraries: torch, transformers, peft, datasets, matplotlib, tensorboard

📊 Training Configuration:
Parameter	Value
Epochs	2
Batch Size	10 (train), 8 (val)
Accumulation	5 steps
Optimizer	AdamW
LoRA Rank	64
LoRA Alpha	32
Dropout	0.05
Target Modules	Linear (in==out)

📦 Output:

    Checkpoints every 250 steps

    Final .pt and safetensors weights

    TensorBoard logs (loss, tokens/sec, cost estimates)

💡 Optimizable Areas:

    Increase max_length if most samples are longer

    Adjust r, alpha, and batch size for stability/performance

    Tune validation frequency or reduce val batch for faster eval

    Add hard negative mining in future versions

🎯 Final Outcome

You get a domain-labeled, high-quality triplet dataset and a fine-tuned LoRA embedding model for semantic memory.

🛠️ Applications:

    Vector DB training (e.g., FAISS, Qdrant)

    Embedding model fine-tuning

    Semantic retrieval tasks

    Memory-aware agents or LLMs

🧠 Summary of Flow:

anchor-builder.py        →   domain anchor generation
vector-bank-generator.py →   triplet generation (anchor + pos + neg)
vector-lora-trainer.py   →   model fine-tuning on (anchor + pos) pairs

📤 Output: 250K+ triplets → LoRA fine-tuned semantic model for vector memory.
