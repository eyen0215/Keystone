"""
Graph triple extraction and metrics computation.
Parses Graph-PReFLexOR's structured output into NetworkX graphs
and computes the metrics from Buehler (2025) for baseline comparison.
"""

import re
import networkx as nx

try:
    import community as community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False


def parse_triples(text):
    """
    Extract (source, target, relation) triples from model output text.

    Graph-PReFLexOR produces graphs in several formats within <|thinking|>:
      1. Arrow notation:  "NodeA" -> "NodeB" [label="RELATIONSHIP"]
      2. Keyword relations: NodeA IS-A NodeB
      3. Dash notation: NodeA -- RELATIONSHIP --> NodeB

    Returns:
        list of (source, target, relation) tuples
    """
    triples = []

    # Pattern 1: Arrow-based (most common in Graph-PReFLexOR output)
    # Handles: "Node A" -> "Node B" [label="REL"]
    # Handles: Node A -> Node B
    for m in re.finditer(
        r'["\']?([^"\'>\n]{2,80}?)["\']?\s*->\s*["\']?([^"\'>\n\[]{2,80}?)["\']?'
        r'\s*(?:\[label=["\']([^"\']*)["\'])?\s*',
        text
    ):
        src = m.group(1).strip().strip('"\'')
        tgt = m.group(2).strip().strip('"\'')
        rel = (m.group(3) or "RELATES-TO").strip()
        if src and tgt and src != tgt:
            triples.append((src, tgt, rel))

    # Pattern 2: Named relationships in prose
    RELATIONS = (
        "IS-A|RELATES-TO|INFLUENCES|PART-OF|CAUSES|ENABLES|"
        "INHIBITS|HAS-PROPERTY|DERIVED-FROM|COMPOSED-OF|LEADS-TO"
    )
    for m in re.finditer(
        rf'["\']?([A-Za-z][A-Za-z\s]{{1,60}}?)["\']?\s+'
        rf'({RELATIONS})\s+'
        rf'["\']?([A-Za-z][A-Za-z\s]{{1,60}}?)["\']?\s*$',
        text, re.MULTILINE
    ):
        triples.append((m.group(1).strip(), m.group(3).strip(), m.group(2).strip()))

    # Pattern 3: Dash notation
    for m in re.finditer(
        r'([A-Za-z][A-Za-z\s]{1,60}?)\s*--\s*([A-Z_-]+)\s*-->\s*([A-Za-z][A-Za-z\s]{1,60})',
        text
    ):
        triples.append((m.group(1).strip(), m.group(3).strip(), m.group(2).strip()))

    # Deduplicate (case-insensitive)
    seen = set()
    unique = []
    for s, t, r in triples:
        key = (s.lower(), t.lower(), r.lower())
        if key not in seen:
            seen.add(key)
            unique.append((s, t, r))

    return unique


def build_graph(triples, existing_graph=None):
    """
    Build or extend a NetworkX DiGraph from triples.

    Args:
        triples: list of (source, target, relation)
        existing_graph: optional existing graph to extend

    Returns:
        NetworkX DiGraph
    """
    G = existing_graph if existing_graph is not None else nx.DiGraph()
    for src, tgt, rel in triples:
        G.add_edge(src, tgt, relation=rel)
    return G


def compute_graph_metrics(G):
    """
    Compute graph metrics matching those reported in Buehler (2025).

    Key metrics from the paper:
      - Modularity: ~0.70 (stable after initial growth)
      - Transitivity: declines from ~0.35 to ~0.10 over 1000 iterations
      - Bridge nodes: persistent, identified by betweenness centrality
      - Scale-free properties: power-law degree distribution

    Returns:
        dict of metric name -> value
    """
    metrics = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": round(nx.density(G), 6),
    }

    if G.number_of_nodes() == 0:
        return metrics

    # --- Clustering & Transitivity ---
    G_undirected = G.to_undirected()

    try:
        metrics["transitivity"] = round(nx.transitivity(G_undirected), 4)
    except Exception:
        metrics["transitivity"] = 0.0

    try:
        metrics["avg_clustering"] = round(nx.average_clustering(G_undirected), 4)
    except Exception:
        metrics["avg_clustering"] = 0.0

    # --- Modularity (Louvain) ---
    if LOUVAIN_AVAILABLE and G.number_of_nodes() > 2:
        try:
            partition = community_louvain.best_partition(G_undirected)
            metrics["modularity"] = round(
                community_louvain.modularity(partition, G_undirected), 4
            )
            metrics["num_communities"] = len(set(partition.values()))
        except Exception:
            metrics["modularity"] = None
            metrics["num_communities"] = None
    else:
        metrics["modularity"] = None
        metrics["num_communities"] = None

    # --- Degree statistics ---
    degrees = [d for _, d in G.degree()]
    metrics["max_degree"] = max(degrees)
    metrics["avg_degree"] = round(sum(degrees) / len(degrees), 2)

    # --- Bridge nodes (betweenness centrality) ---
    if G.number_of_nodes() > 2:
        try:
            bc = nx.betweenness_centrality(G)
            sorted_bc = sorted(bc.items(), key=lambda x: -x[1])
            metrics["top_bridge_nodes"] = [
                {"node": n, "betweenness": round(v, 4)}
                for n, v in sorted_bc[:5]
            ]
        except Exception:
            metrics["top_bridge_nodes"] = []
    else:
        metrics["top_bridge_nodes"] = []

    # --- Connected components (directed) ---
    try:
        metrics["num_weakly_connected"] = nx.number_weakly_connected_components(G)
        metrics["num_strongly_connected"] = nx.number_strongly_connected_components(G)
    except Exception:
        metrics["num_weakly_connected"] = None
        metrics["num_strongly_connected"] = None

    return metrics
