"""
01_load_and_infer.py — Verify model loading + single inference
==============================================================
Run this first to confirm everything is installed correctly.

Usage:
    python src/01_load_and_infer.py
"""

import os
import sys
import json
import time

# Add project root to path so utils imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.generation import load_model, generate_response, split_thinking_answer
from src.utils.graph_parsing import parse_triples, compute_graph_metrics, build_graph

# =============================================================================
# CONFIG
# =============================================================================
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

QUESTION = (
    "Discuss how hierarchical structures in biological materials, "
    "like spider silk and bone, can inspire the design of new "
    "tough composite materials."
)

# =============================================================================
# LOAD
# =============================================================================
model, tokenizer, info = load_model()
print(f"\nModel info: {json.dumps(info, indent=2)}\n")

# =============================================================================
# INFER
# =============================================================================
print(f"Question: {QUESTION[:80]}...")
print("Generating...\n")

t0 = time.time()
raw_output = generate_response(
    model, tokenizer,
    prompt=f"{QUESTION} Use <|thinking|>.",
    max_new_tokens=2048,
    temperature=0.3,
)
gen_time = time.time() - t0

# =============================================================================
# PARSE
# =============================================================================
thinking, answer = split_thinking_answer(raw_output)
triples = parse_triples(thinking)
G = build_graph(triples)
metrics = compute_graph_metrics(G)

# =============================================================================
# REPORT
# =============================================================================
print(f"{'='*60}")
print(f"RESULTS")
print(f"{'='*60}")
print(f"Generation time:  {gen_time:.1f}s")
print(f"Thinking length:  {len(thinking)} chars")
print(f"Answer length:    {len(answer)} chars")
print(f"Triples found:    {len(triples)}")
print(f"Graph:            {metrics['num_nodes']} nodes, {metrics['num_edges']} edges")

if triples:
    print(f"\nExtracted triples:")
    for s, t, r in triples[:15]:
        print(f"  {s} --[{r}]--> {t}")

if not triples:
    print(f"\nNo triples parsed. This might mean the model output uses a format")
    print(f"not yet covered by the parser. Check the raw output below and in")
    print(f"outputs/raw_output.txt to see what format the model is using.\n")

# =============================================================================
# SAVE
# =============================================================================
results = {
    "question": QUESTION,
    "generation_time_s": round(gen_time, 1),
    "model_info": info,
    "thinking_section": thinking,
    "answer_section": answer,
    "triples": [{"source": s, "target": t, "relation": r} for s, t, r in triples],
    "graph_metrics": metrics,
}

with open(os.path.join(OUTPUT_DIR, "inference_result.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

with open(os.path.join(OUTPUT_DIR, "raw_output.txt"), "w", encoding="utf-8") as f:
    f.write(f"QUESTION: {QUESTION}\n\n")
    f.write(f"{'='*60}\nFULL RAW OUTPUT:\n{'='*60}\n")
    f.write(raw_output)
    f.write(f"\n\n{'='*60}\nTHINKING SECTION:\n{'='*60}\n")
    f.write(thinking)
    f.write(f"\n\n{'='*60}\nANSWER SECTION:\n{'='*60}\n")
    f.write(answer)

print(f"\nSaved to: {OUTPUT_DIR}/inference_result.json")
print(f"Saved to: {OUTPUT_DIR}/raw_output.txt")

# Preview
print(f"\n{'='*60}")
print("ANSWER PREVIEW:")
print(f"{'='*60}")
preview = answer[:500] if answer else thinking[:500]
print(preview)
if len(preview) >= 500:
    print("...")

print(f"\n✓ Model is working. Next: python src/02_iterative_graph_growth.py")
