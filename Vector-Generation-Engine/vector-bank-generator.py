# === Start Server ====
#CUDA_VISIBLE_DEVICES=0 ~/Desktop/Vector-DB-LoRA/llama.cpp/build/bin/llama-server \
#  -m /home/ndesantis/Desktop/Vector-DB-LoRA/Meta-Llama-3-8B-Instruct.Q6_K.gguf \
#  --port 8080 \
#  --ctx-size 1792 \
#  --n-gpu-layers 256


#CUDA_VISIBLE_DEVICES=0 ~/Desktop/Vector-DB-LoRA/llama.cpp/build/bin/llama-server \
#  -m /home/ndesantis/Desktop/Vector-DB-LoRA/Meta-Llama-3-8B-Instruct.Q6_K.gguf \
#  --port 8082 \
#  --ctx-size 1792 \
#  --n-gpu-layers 256

#CUDA_VISIBLE_DEVICES=1 ~/Desktop/Vector-DB-LoRA/llama.cpp/build/bin/llama-server \
#  -m /home/ndesantis/Desktop/Vector-DB-LoRA/Meta-Llama-3-8B-Instruct.Q6_K.gguf \
#  --port 8081 \
#  --ctx-size 1792 \
#  --n-gpu-layers 256

#CUDA_VISIBLE_DEVICES=1 ~/Desktop/Vector-DB-LoRA/llama.cpp/build/bin/llama-server \
#  -m /home/ndesantis/Desktop/Vector-DB-LoRA/Meta-Llama-3-8B-Instruct.Q6_K.gguf \
#  --port 8083 \
#  --ctx-size 1792 \
#  --n-gpu-layers 256


# === Run Script Command ===
#cd /home/ndesantis/Desktop/Vector-DB-LoRA/JSONL-Generation-Engine
#python3 vector-bank-generator.py --count 250000 --threads 12 --anchors ./content --analyze


import json
import time
import random
import argparse
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import threading
import queue

triplet_buffer = []
buffer_lock = threading.Lock()
def buffered_writer(path):
    with open(path, "a", encoding="utf-8") as fout:
        while True:
            time.sleep(0.5)
            with buffer_lock:
                if triplet_buffer:
                    for triplet in triplet_buffer:
                        fout.write(json.dumps(triplet, ensure_ascii=False) + "\n")
                    triplet_buffer.clear()


try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    USE_TQDM = False

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

from collections import defaultdict

API_PORTS = [8080, 8081, 8082, 8083]
API_LOAD = defaultdict(int)

def get_least_loaded_url():
    return f"http://localhost:{min(API_PORTS, key=lambda port: API_LOAD[port])}/v1/chat/completions"


HEADERS = {"Content-Type": "application/json"}
MAX_RETRIES = 2
DEFAULT_OUTPUT_DIR = "./content"

DOMAIN_TARGETS = {
    "reflection": 35000,
    "goals": 35000,
    "task": 35000,
    "observation": 35000,
    "tool": 35000,
    "knowledge": 35000,
    "chat": 40000
}

FALLBACK_SEED_ANCHORS = [
    {"domain": "reflection", "text": "I'm feeling overwhelmed by all the tasks on my plate."},
    {"domain": "goals", "text": "I need to remember to submit that report before noon."},
    {"domain": "task", "text": "I should follow up with the doctor next week."},
    {"domain": "observation", "text": "That meeting left me feeling unsure and anxious."},
    {"domain": "tool", "text": "Set a reminder to check the budget spreadsheet."},
    {"domain": "knowledge", "text": "Vitamin D deficiency can cause fatigue."},
    {"domain": "chat", "text": "Do you remember what I told you yesterday?"}
]

global_max = 0

def make_batched_prompt(anchor_texts):
    numbered = "\n".join(
        [f'{i+1}. "{text}"' for i, text in enumerate(anchor_texts)]
    )
    return f"""
You are an AI memory assistant helping build a semantic memory system.

Generate triplets for the following anchors:

{numbered}

For each anchor, generate:
1. A semantically similar version (positive)
2. A semantically unrelated or misleading version (negative)

Respond in this exact JSON array format:
[
  {{
    "anchor": "...",
    "positive": "...",
    "negative": "..."
  }},
  ...
]
"""


def extract_json(content):
    start = content.find('[')
    end = content.rfind(']')
    if start != -1 and end != -1 and end > start:
        snippet = content[start:end+1]
        try:
            return json.loads(snippet)
        except Exception as e:
            print(f"⚠️ JSON extraction failed: {e}")
    return None

def get_triplet_batch(anchor_batch, _):  # ignore incoming api_url param
    api_url = get_least_loaded_url()
    port = int(api_url.split(":")[2].split("/")[0])
    API_LOAD[port] += 1  # mark port as busy

    prompt = make_batched_prompt([a["text"] for a in anchor_batch])
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 512  # adjust if needed
    }

    try:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.post(api_url, json=payload, headers=HEADERS, timeout=60)
                if response.status_code == 200:
                    content = response.json()["choices"][0]["message"]["content"].strip()
                    try:
                        parsed = json.loads(content)
                    except:
                        parsed = extract_json(content)

                    if isinstance(parsed, list) and all("anchor" in p and "positive" in p and "negative" in p for p in parsed):
                        results = []
                        for anchor, triplet in zip(anchor_batch, parsed):
                            triplet["domain"] = anchor["domain"]
                            combined = f'Anchor: "{triplet["anchor"]}"\nPositive: {triplet["positive"]}\nNegative: {triplet["negative"]}'
                            tokens_used = len(tokenizer(combined)["input_ids"])
                            triplet["token_length"] = tokens_used
                            results.append(triplet)

                            global global_max
                            if tokens_used > global_max:
                                global_max = tokens_used
                                print(f"🚀 NEW MAX TOKENS: {global_max}")

                            print(f"🧮 {anchor['domain']} | Tokens: {tokens_used}")
                        return results
                    else:
                        print(f"⚠️ Invalid batch JSON (attempt {attempt}): {content[:200]}")
                else:
                    print(f"❌ HTTP {response.status_code}: {response.text[:120]}")
            except Exception as e:
                print(f"⚠️ Exception on attempt {attempt}: {e}")
            time.sleep(1 + random.uniform(0, 1))
        return None

    finally:
        API_LOAD[port] -= 1  # always decrement port load


def load_all_domain_anchors(directory):
    anchors_by_domain = {}
    for filename in os.listdir(directory):
        if filename.endswith(".jsonl"):
            domain = filename.replace(".jsonl", "")
            path = os.path.join(directory, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    anchors = []
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            if "text" in obj:
                                obj["domain"] = domain
                                anchors.append(obj)
                        except:
                            continue
                    if anchors:
                        anchors_by_domain[domain] = anchors
                        print(f"📥 Loaded {len(anchors)} from {filename}")
            except Exception as e:
                print(f"⚠️ Error reading {filename}: {e}")
    return anchors_by_domain

def analyze_lengths(jsonl_path):
    print("📏 Analyzing token lengths...")
    lengths = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            prompt = f'Anchor:\n"{entry["anchor"]}"\n\nGenerate similar and dissimilar thoughts.\n\n'
            response = f'Positive: {entry["positive"]}\nNegative: {entry["negative"]}'
            combined = prompt + response
            lengths.append(len(tokenizer(combined)["input_ids"]))

    plt.hist(lengths, bins=50)
    plt.title("Token Lengths (Prompt + Positive + Negative)")
    plt.xlabel("Number of tokens")
    plt.ylabel("Count")
    plt.grid(True)
    plt.show()

    print(f"\n📊 Token Length Stats:")
    print(f"Max: {max(lengths)}")
    print(f"Avg: {sum(lengths) // len(lengths)}")
    print(f"95th percentile: {sorted(lengths)[int(0.95 * len(lengths))]}")
    print(f"99th percentile: {sorted(lengths)[int(0.99 * len(lengths))]}")

def main():
    parser = argparse.ArgumentParser(description="Generate vector triplet training data using LLaMA")
    parser.add_argument("--count", type=int, default=250000, help="Number of triplets to generate")
    parser.add_argument("--threads", type=int, default=24, help="Number of parallel threads")
    parser.add_argument("--anchors", type=str, help="Path to folder with per-domain anchors")
    parser.add_argument("--output", type=str, help="Custom output filename (.jsonl)")
    parser.add_argument("--analyze", action="store_true", help="Run token length analysis after generation")
    args = parser.parse_args()

    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output or os.path.join(DEFAULT_OUTPUT_DIR, f"vector_triplets_{timestamp}.jsonl")

    if args.anchors and os.path.isdir(args.anchors):
        anchors_by_domain = load_all_domain_anchors(args.anchors)
    else:
        print("⚠️ No valid anchor directory provided. Using fallback anchors.")
        anchors_by_domain = {}
        for anchor in FALLBACK_SEED_ANCHORS:
            domain = anchor["domain"]
            anchors_by_domain.setdefault(domain, []).append(anchor)
    from copy import deepcopy

# Shuffle and keep backup of full anchor pool
    for domain, anchors in anchors_by_domain.items():
        random.shuffle(anchors)
    original_anchors = deepcopy(anchors_by_domain)

    domain_counts = {d: 0 for d in DOMAIN_TARGETS}
    total_target = sum(DOMAIN_TARGETS.values())
    max_count = min(args.count, total_target)

    print(f"\n🔁 Generating up to {max_count} triplets using {args.threads} threads...")
    print(f"💾 Output: {output_file}\n")

    start_time = time.time()
        # Start background writer thread
    writer_thread = threading.Thread(target=buffered_writer, args=(output_file,), daemon=True)
    writer_thread.start()

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = []
        completed = 0

        while sum(domain_counts.values()) < max_count:
            for domain, target in DOMAIN_TARGETS.items():
                if domain_counts[domain] >= target or domain not in anchors_by_domain:
                    continue

                # Refill and reshuffle if we run out
                if len(anchors_by_domain[domain]) < 6:
                    anchors_by_domain[domain] = original_anchors[domain][:]
                    random.shuffle(anchors_by_domain[domain])

                # Take 6 anchors and trim the list
                anchor_batch = anchors_by_domain[domain][:6]
                anchors_by_domain[domain] = anchors_by_domain[domain][6:]

                futures.append(executor.submit(get_triplet_batch, anchor_batch, None))
                domain_counts[domain] += 6

                if sum(domain_counts.values()) >= max_count:
                    break


            if USE_TQDM:
                progress = tqdm(total=len(futures), desc="⏳ Waiting on completions", leave=False)

            for idx, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result:
                    for triplet in result:
                        with buffer_lock:
                            triplet_buffer.append(triplet)
                        completed += 1

                elapsed = time.time() - start_time
                if completed % 100 == 0:
                    rate = completed / elapsed
                    eta = (max_count - completed) / rate
                    print(f"✅ {completed}/{max_count} | Speed: {rate:.2f} triplets/sec | ETA: {eta/60:.1f} min")

                if USE_TQDM:
                    progress.update(1)

            if USE_TQDM:
                progress.close()

            futures = []


    duration = time.time() - start_time
    print(f"\n✅ All done. Output saved to: {output_file}")
    print(f"📊 Final domain counts:\n{json.dumps(domain_counts, indent=2)}")
    print(f"⏱️ Total time: {duration:.2f} sec | Avg speed: {completed / duration:.2f} triplets/sec")
    print(f"📈 Max token length observed: {global_max}")

    if args.analyze:
        analyze_lengths(output_file)

if __name__ == "__main__":
    main()
