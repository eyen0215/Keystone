# Graph-PReFLexOR: Constraint Auditing & Recursive Saliency Pruning

Research project extending [Buehler (2025)](https://arxiv.org/abs/2501.08120)'s Graph-PReFLexOR framework with:
- **Constraint Auditing Layers** — trajectory-level validation to detect semantic drift
- **Recursive Saliency Pruning** — structure-aware sparsification for emergent knowledge graphs

Built on the [Graph-PReFLexOR](https://huggingface.co/lamm-mit/Graph-Preflexor_01062025) model (3B params, fine-tuned Llama 3.2-3B-Instruct).

## Quick Start

### Prerequisites
- Python 3.10+
- NVIDIA GPU with ≥8GB VRAM (tested on RTX 5070 Ti 16GB)
- Git

### Setup

```bash
# 1. Clone this repo
git clone https://github.com/<your-username>/graph-preflexor-project.git
cd graph-preflexor-project

# 2. Create virtual environment
python -m venv venv

# Windows:
.\venv\Scripts\activate
# Linux/Mac:
# source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** First run of any script downloads the model (~6GB) from HuggingFace.
> This is cached after the first download.

### Run

```bash
# Step 1: Verify model loads and produces output (~2 min after download)
python src/01_load_and_infer.py

# Step 2: Run baseline knowledge garden growth (~20-40 min for 20 iterations)
python src/02_iterative_graph_growth.py

# Step 3: (coming soon) Error injection + constraint auditing prototype
```

## Project Structure

```
graph-preflexor-project/
├── README.md
├── requirements.txt
├── setup.cfg
├── .gitignore
├── configs/
│   └── default.yaml          # Experiment configuration
├── src/
│   ├── 01_load_and_infer.py  # Test model loading + single inference
│   ├── 02_iterative_graph_growth.py  # Baseline knowledge garden + metrics
│   └── utils/
│       ├── __init__.py
│       ├── generation.py      # Model loading & inference wrappers
│       └── graph_parsing.py   # Triple extraction & graph metrics
├── notebooks/                 # Jupyter notebooks for exploration
├── outputs/                   # Generated graphs, plots, logs (gitignored)
└── tests/                     # Unit tests (future)
```

## Key Metrics Tracked

| Metric | Paper Value | Our Target |
|--------|------------|------------|
| Modularity (Louvain) | ~0.70 stable | Reproduce within 10% |
| Transitivity | 0.35 → 0.10 decline | Observe same trend |
| Bridge node persistence | Stable hubs emerge late | Identify & track |
| Scale-free degree dist. | Power-law | Verify with log-log plot |

## References

```bibtex
@article{buehler2025GraphPRefLexOR,
    title={In-situ graph reasoning and knowledge expansion using Graph-PReFLexOR},
    author={Markus J. Buehler},
    year={2025},
    eprint={2501.08120},
    archivePrefix={arXiv},
    primaryClass={cs.AI},
}
```
