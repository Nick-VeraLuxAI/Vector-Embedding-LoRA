# === Start Server Command ===
#/home/ndesantis/Desktop/Vector-DB-LoRA/llama.cpp/build/bin/llama-server \
#  --model "/home/ndesantis/Desktop/Vector-DB-LoRA/Meta-Llama-3-8B-Instruct.Q6_K.gguf" \
#  --port 8080 \
#  --threads 12 \
#  --n-gpu-layers 100

# === Run Command ===
#python3 /home/ndesantis/Desktop/Vector-DB-LoRA/JSONL-Generation-Engine/anchor-builder.py

import json
import time
import random
import os
import argparse
import requests
from datetime import datetime
import multiprocessing
import matplotlib.pyplot as plt

API_URL = "http://localhost:8080/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}
MAX_RETRIES = 3
DELAY = 1.2  # seconds between retries
OUTPUT_DIR = "/home/ndesantis/Desktop/Vector-DB-LoRA/JSONL-Generation-Engine/content"
MAX_TOKENS_ALLOWED = 100
DOMAIN_CONFIG = {
    "reflection": 1000,
    "goals": 1000,
    "task": 1000,
    "observation": 1000,
    "tool": 1000,
    "knowledge": 1000,
    "chat": 1200
}

SEED_EXAMPLES = {
    "reflection": "I'm feeling overwhelmed by everything going on.",
    "goals": "I want to finish my portfolio by next week.",
    "task": "I need to call the mechanic tomorrow morning.",
    "observation": "The room went silent when I brought up the issue.",
    "tool": "Set a reminder to check the budget spreadsheet.",
    "knowledge": "Water boils at 100°C under standard conditions.",
    "chat": "Can you remind me what I said about the deadline?"
}

def make_prompt(domain, example):
    return f"""You are helping build a semantic memory system for the domain of \"{domain}\".\n\nGiven this example:\n\"{example}\"\n\nGenerate exactly 20 semantically diverse, realistic thoughts in the \"{domain}\" category.\n\nRespond ONLY with a valid JSON array of strings (no explanation, no list formatting):\n[\n  \"...\",\n  \"...\",\n  ...\n]\n"""

def call_model(prompt):
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "max_tokens": 800
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(API_URL, json=payload, headers=HEADERS, timeout=60)
            if response.status_code == 200:
                raw = response.json()["choices"][0]["message"]["content"]
                try:
                    return json.loads(raw)
                except Exception as e:
                    print(f"\n❌ JSON parsing failed. Model output:\n{raw}\n--- Error: {e}")
            else:
                print(f"❌ HTTP {response.status_code}: {response.text[:80]}")
        except Exception as e:
            print(f"⚠️ Exception on attempt {attempt+1}: {e}")
        time.sleep(DELAY + random.uniform(0, 0.5))
    return []

def analyze_threads():
    total_cores = multiprocessing.cpu_count()
    logical_threads = os.cpu_count()
    print(f"\n🧠 System Thread Analyzer")
    print(f"🧵 Logical CPU Threads: {logical_threads}")
    print(f"💡 Recommended Threads for Heavy API Use: {int(logical_threads * 0.75)} or fewer\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_request", type=int, default=20, help="Anchors per API call")
    parser.add_argument("--target", type=int, help="Override per-domain target")
    args = parser.parse_args()

    analyze_threads()
    lengths = []

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        for domain, target in DOMAIN_CONFIG.items():
            goal = args.target or target
            out_path = os.path.join(OUTPUT_DIR, f"{domain}.jsonl")
            collected = 0
            print(f"\n🔧 Generating {goal} anchors for domain: {domain}")
            with open(out_path, "a", encoding="utf-8") as f:
                while collected < goal:
                    example = SEED_EXAMPLES[domain]
                    prompt = make_prompt(domain, example)
                    batch = call_model(prompt)
                    if isinstance(batch, list):
                        lengths.append(len(prompt.split()) + sum(len(x.split()) for x in batch if isinstance(x, str)))
                        for a in batch:
                            cleaned = a.strip()
                            if not cleaned:
                                continue
                            token_len = len(cleaned.split())
                            if token_len > MAX_TOKENS_ALLOWED:
                                print(f"⚠️ Trimming long anchor ({token_len} tokens): {cleaned[:60]}...")
                                cleaned = " ".join(cleaned.split()[:MAX_TOKENS_ALLOWED])
                            f.write(json.dumps({"domain": domain, "text": cleaned}, ensure_ascii=False) + "\n")
                            collected += 1
                        print(f"[{domain}] {collected}/{goal} collected...")
                    else:
                        print(f"⚠️ Failed to parse batch: {batch}")
    finally:
        if lengths:
            plt.hist(lengths, bins=50)
            plt.title("Prompt+Response Token Lengths")
            plt.xlabel("Number of tokens")
            plt.ylabel("Count")
            plt.show()

            print("Max length:", max(lengths))
            print("Average length:", sum(lengths) // len(lengths))
            print("95th percentile:", sorted(lengths)[int(0.95 * len(lengths))])
            print("99th percentile:", sorted(lengths)[int(0.99 * len(lengths))])

if __name__ == "__main__":
    main()
