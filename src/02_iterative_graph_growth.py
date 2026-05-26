"""
02_iterative_graph_growth.py — Baseline Knowledge Garden + Metrics
===================================================================
Runs a small-scale version of Buehler's "knowledge garden" algorithm
and tracks the same metrics reported in the paper.

This is your Month 1 deliverable: "reproduce baseline results."

Usage:
    python src/02_iterative_graph_growth.py
    python src/02_iterative_graph_growth.py --iterations 50
"""

import os
import sys
import json
import time
import random
import argparse
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (saves to file)
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.generation import load_model, generate_response, split_thinking_answer
from src.utils.graph_parsing import parse_triples, build_graph, compute_graph_metrics


# =============================================================================
# CLI ARGS
# =============================================================================
parser = argparse.ArgumentParser(description="Knowledge Garden Growth")
parser.add_argument("--iterations", type=int, default=20, help="Number of growth iterations")
parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature")
parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens per generation")
parser.add_argument("--seed-question", type=str,
                    default="Discuss an interesting idea in bio-inspired materials science.",
                    help="Initial seed question")
parser.add_argument("--output-dir", type=str, default="./outputs/garden", help="Output directory")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)


# =============================================================================
# LOAD MODEL
# =============================================================================
model, tokenizer, model_info = load_model()


# =============================================================================
# KNOWLEDGE GARDEN LOGIC
# =============================================================================
def generate_followup_question(G, fallback_question):
    """Generate a follow-up question from graph state (self-questioning)."""
    if G.number_of_nodes() < 2:
        return fallback_question

    degrees = dict(G.degree())
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:3]
    random_node = random.choice(list(G.nodes()))

    return (
        f"Based on the concepts '{top_nodes[0]}' and '{random_node}', "
        f"propose a new research question that connects these ideas in an "
        f"innovative way, especially relating to materials science or engineering."
    )


# =============================================================================
# MAIN LOOP
# =============================================================================
print(f"\n{'='*60}")
print(f"KNOWLEDGE GARDEN GROWTH")
print(f"  Iterations:  {args.iterations}")
print(f"  Temperature: {args.temperature}")
print(f"  Seed:        {args.seed_question[:60]}...")
print(f"  Output:      {args.output_dir}")
print(f"{'='*60}\n")

G = nx.DiGraph()
metrics_history = []
all_triples = []
iteration_log = []
current_question = args.seed_question

total_start = time.time()

for i in range(args.iterations):
    iter_start = time.time()
    print(f"[{i+1}/{args.iterations}] {current_question[:70]}...")

    # Generate
    raw_output = generate_response(
        model, tokenizer,
        prompt=f"{current_question} Use <|thinking|>.",
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # Parse
    thinking, answer = split_thinking_answer(raw_output)
    new_triples = parse_triples(thinking)
    G = build_graph(new_triples, existing_graph=G)
    all_triples.extend(new_triples)

    # Metrics
    metrics = compute_graph_metrics(G)
    metrics["iteration"] = i + 1
    metrics["new_triples"] = len(new_triples)
    metrics["cumulative_triples"] = len(all_triples)
    metrics_history.append(metrics)

    iter_time = time.time() - iter_start
    mod_str = f"{metrics['modularity']:.3f}" if metrics.get('modularity') is not None else "N/A"
    print(f"         +{len(new_triples)} triples | {metrics['num_nodes']}n/{metrics['num_edges']}e | "
          f"mod={mod_str} trans={metrics['transitivity']:.3f} | {iter_time:.1f}s")

    iteration_log.append({
        "iteration": i + 1,
        "question": current_question,
        "new_triples": len(new_triples),
        "time_s": round(iter_time, 1),
        "metrics": metrics,
    })

    # Next question
    current_question = generate_followup_question(G, args.seed_question)

total_time = time.time() - total_start


# =============================================================================
# SAVE DATA
# =============================================================================
print(f"\nSaving results...")

nx.write_graphml(G, os.path.join(args.output_dir, "knowledge_graph.graphml"))

with open(os.path.join(args.output_dir, "metrics_history.json"), "w") as f:
    json.dump(metrics_history, f, indent=2, default=str)

with open(os.path.join(args.output_dir, "iteration_log.json"), "w") as f:
    json.dump(iteration_log, f, indent=2, default=str)

with open(os.path.join(args.output_dir, "all_triples.json"), "w") as f:
    json.dump([{"source": s, "target": t, "relation": r} for s, t, r in all_triples], f, indent=2)


# =============================================================================
# PLOTS
# =============================================================================
if len(metrics_history) > 1:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Knowledge Garden Growth — Baseline Metrics", fontsize=14, fontweight="bold")

    iters = [m["iteration"] for m in metrics_history]

    # Graph size
    axes[0, 0].plot(iters, [m["num_nodes"] for m in metrics_history], "b-o", ms=3, label="Nodes")
    axes[0, 0].plot(iters, [m["num_edges"] for m in metrics_history], "r-o", ms=3, label="Edges")
    axes[0, 0].set_title("Graph Size"); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    # Transitivity
    axes[0, 1].plot(iters, [m["transitivity"] for m in metrics_history], "g-o", ms=3)
    axes[0, 1].axhline(0.35, color="gray", ls="--", alpha=0.5, label="Paper start (~0.35)")
    axes[0, 1].axhline(0.10, color="gray", ls=":", alpha=0.5, label="Paper end (~0.10)")
    axes[0, 1].set_title("Transitivity"); axes[0, 1].legend(fontsize=7); axes[0, 1].grid(True, alpha=0.3)

    # Modularity
    mv = [(m["iteration"], m["modularity"]) for m in metrics_history if m["modularity"] is not None]
    if mv:
        axes[0, 2].plot([x[0] for x in mv], [x[1] for x in mv], "m-o", ms=3)
        axes[0, 2].axhline(0.70, color="gray", ls="--", alpha=0.5, label="Paper (~0.70)")
        axes[0, 2].legend(fontsize=7)
    axes[0, 2].set_title("Modularity (Louvain)"); axes[0, 2].grid(True, alpha=0.3)

    # New triples per iteration
    axes[1, 0].bar(iters, [m["new_triples"] for m in metrics_history], color="steelblue", alpha=0.7)
    axes[1, 0].set_title("New Triples / Iteration"); axes[1, 0].grid(True, alpha=0.3)

    # Density
    axes[1, 1].plot(iters, [m["density"] for m in metrics_history], "orange", marker="o", ms=3)
    axes[1, 1].set_title("Density"); axes[1, 1].grid(True, alpha=0.3)

    # Avg degree
    axes[1, 2].plot(iters, [m["avg_degree"] for m in metrics_history], "brown", marker="o", ms=3)
    axes[1, 2].set_title("Avg Degree"); axes[1, 2].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("Iteration")

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "growth_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close()

# Graph visualization
if G.number_of_nodes() > 0:
    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(G, k=2.0, iterations=50, seed=42) if G.number_of_nodes() < 150 \
        else nx.kamada_kawai_layout(G)
    degrees = dict(G.degree())
    sizes = [max(degrees[n] * 80, 100) for n in G.nodes()]
    nx.draw_networkx_edges(G, pos, alpha=0.15, arrows=True, arrowsize=8, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color="steelblue", alpha=0.7, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=6, ax=ax)
    ax.set_title(f"Knowledge Garden ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    ax.axis("off")
    plt.savefig(os.path.join(args.output_dir, "knowledge_graph.png"), dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# SUMMARY
# =============================================================================
print(f"\n{'='*60}")
print(f"DONE — {total_time:.0f}s ({total_time/60:.1f} min)")
print(f"{'='*60}")
if metrics_history:
    final = metrics_history[-1]
    print(f"  Final graph:   {final['num_nodes']} nodes, {final['num_edges']} edges")
    print(f"  Transitivity:  {final['transitivity']:.4f}  (paper: 0.35→0.10)")
    mod = final.get("modularity")
    print(f"  Modularity:    {f'{mod:.4f}' if mod else 'N/A'}  (paper: ~0.70)")
    print(f"  Avg degree:    {final['avg_degree']:.2f}")
    if final.get("top_bridge_nodes"):
        print(f"  Top bridges:   {', '.join(b['node'] for b in final['top_bridge_nodes'][:3])}")
print(f"\n  Files in: {args.output_dir}/")
print(f"    knowledge_graph.graphml  — loadable in NetworkX / Gephi")
print(f"    knowledge_graph.png      — visual")
print(f"    growth_metrics.png       — 6-panel metric trends")
print(f"    metrics_history.json     — raw data for analysis")
print(f"\n  Next: examine raw_output.txt, tune parser if needed, try --iterations 50")
