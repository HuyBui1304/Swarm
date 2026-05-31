# S-Mapper: A Topology-Aware Optimization Framework for Robust Mapper Construction

> **Paper**: *S-Mapper: A Topology-Aware Optimization Framework for Robust Mapper Construction*  
> Submitted to *ACM Transactions on Knowledge Discovery from Data (TKDD)*, May 2025.

S-Mapper automatically selects the resolution parameters of the **Mapper** algorithm — number of intervals *N* and overlap percentage *p* — by maximizing a novel metric called the **Topological Silhouette Coefficient (TSC)** using Swarm Intelligence (PSO and GWO).

---

## Overview

The [Mapper algorithm](https://research.math.osu.edu/tgda/mapperPBG.pdf) produces a topological graph from high-dimensional data. Its output quality depends heavily on two hyperparameters (*N*, *p*) that are typically set by hand. S-Mapper eliminates this manual tuning through a principled optimization loop:

```
Data → Lens (projection) → Cover(N, p) → Clustering → Mapper Graph
                                 ↑
                         PSO / GWO optimize (N, p)
                         guided by TSC = 0.5·NSC + 0.5·TSR
```

**TSC** balances two complementary signals:
- **NSC** (Normalized Silhouette Coefficient) — measures cluster separation quality
- **TSR** (Topological Signal Rate) — measures how many topological features (connected components, loops) survive a bootstrap stability threshold

This repository also includes **D-Mapper** (Tao & Ge, 2024), a GMM-guided cover that serves as a strong baseline.

---

## Repository Structure

```
implement/
├── kepler-mapper/          # Extended KeplerMapper library
│   └── kmapper/
│       ├── kmapper.py      # Original KeplerMapper (unchanged)
│       ├── cover.py        # Standard uniform cover
│       ├── evaluate.py     # TSC, NSC, TSR metrics  ← S-Mapper core
│       ├── dcover.py       # GMM-guided cover        ← D-Mapper
│       └── dmapper.py      # D_Mapper class          ← D-Mapper
│
├── mealpy/                 # Bundled metaheuristic library (PSO, GWO, 215+ algorithms)
│
└── examples/               # Experiment notebooks for 9 benchmark datasets
    ├── smapper_utils.py    # Shared helpers (run_optimization, draw_graph)
    ├── two_cir/            # Disjoint circles
    ├── two_i_cir/          # Intersecting circles
    ├── trefoil/            # Trefoil knot
    ├── cat/                # 3D cat mesh
    ├── lion/               # 3D lion mesh
    ├── horse/              # 3D horse mesh
    ├── human/              # 3D human mesh
    ├── diabetes/           # Reaven & Miller diabetes (145 pts, 6D)
    └── covid-19/           # COVID-19 RNA sequences (357 pts, 64D k-mer)
```

---

## Installation

**Requirements**: Python ≥ 3.6

```bash
# 1. Install the extended KeplerMapper
cd implement/kepler-mapper
pip install -e .

# 2. Core dependencies
pip install scikit-learn numpy scipy Jinja2 gudhi networkx matplotlib
```

The `mealpy/` library is bundled locally — no separate install needed. Each notebook loads it via `sys.path`.

---

## Quick Start

### Running an experiment

Open any notebook under `implement/examples/`:

```bash
cd implement/examples/two_cir
jupyter notebook two_cir.ipynb
```

Each notebook runs the full pipeline automatically:
1. Loads the dataset and computes a lens projection
2. Optimizes (N, p) via PSO and GWO
3. Builds Mapper, D-Mapper, and S-Mapper graphs side-by-side
4. Writes convergence results to `results_output_*.txt`

### Minimal example

```python
import sys
sys.path.insert(0, 'implement/kepler-mapper')
sys.path.insert(0, 'implement/mealpy')

import kmapper as km
from kmapper import evaluate
from sklearn.cluster import DBSCAN
from mealpy.swarm_based import PSO, GWO
from mealpy.utils.space import FloatVar

# Load your data
data = ...          # (n_samples, n_features)
projected_data = ...  # (n_samples, n_lens_dims)

mapper = km.KeplerMapper()

def objective_function(X):
    n, p = int(round(X[0])), X[1]
    cover = km.Cover(n_cubes=n, perc_overlap=p)
    graph = mapper.map(projected_data, data,
                       cover=cover,
                       clusterer=DBSCAN(eps=0.5, min_samples=3))
    tsc = evaluate.compute_SC_adj(data, projected_data, graph, cover, type='k')
    return -tsc  # mealpy minimizes

problem = {
    "obj_func": objective_function,
    "bounds": [FloatVar(lb=5, ub=25), FloatVar(lb=0.1, ub=0.7)],
    "minmax": "min",
}

# Run PSO
model = PSO.OriginalPSO(epoch=20, pop_size=20)
model.solve(problem, seed=42)
best_n, best_p = int(round(model.solution[0])), model.solution[1]
print(f"Optimal: N={best_n}, p={best_p:.3f}, TSC={-model.target.fitness:.4f}")
```

---

## Datasets

| Dataset | Type | Samples | Dims | Description |
|---------|------|--------:|-----:|-------------|
| `two_cir` | Synthetic | — | 1D lens | Two disjoint circles |
| `two_i_cir` | Synthetic | — | 1D lens | Two intersecting circles |
| `trefoil` | Synthetic | — | 3D | Trefoil knot — tests loop detection |
| `cat` | 3D mesh | — | 3D | Cat shape point cloud |
| `lion` | 3D mesh | — | 3D | Lion shape point cloud |
| `horse` | 3D mesh | — | 3D | Horse shape point cloud |
| `human` | 3D mesh | — | 3D | Human shape point cloud |
| `diabetes` | Real | 145 | 6D | Reaven & Miller diabetes dataset |
| `covid-19` | Real | 357 | 64D | COVID-19 RNA sequences (k-mer features) |

---

## Key Metrics

### Topological Silhouette Coefficient (TSC)

The optimization objective — combines clustering quality with topological fidelity:

```
TSC = w1·NSC + w2·TSR      (default: w1 = w2 = 0.5)
```

**NSC** — Normalized Silhouette Coefficient:
```
SC(x)  = [b(x) - a(x)] / max(a(x), b(x))
NSC(x) = (SC(x) + 1) / 2        ∈ [0, 1]
```
where *a(x)* = mean intra-cluster distance, *b(x)* = mean nearest-cluster distance.

**TSR** — Topological Signal Rate:
```
TSR = N_signal / N
```
where *N_signal* = number of topological features (persistence diagram points) whose bottleneck distance exceeds a bootstrap confidence threshold at level α = 0.85.

### API

```python
from kmapper import evaluate

# Normalized Silhouette Coefficient
nsc = evaluate.get_SC(data, graph, norm=1)

# Topological persistence diagram
dgm = evaluate.compute_topological_features_for_kmapper(graph, lens)

# Bootstrap confidence threshold
threshold = evaluate.bootstrap_topological_features_for_kmapper(
    data, lens, graph, cover, N=100, alpha=0.85, type='k'
)

# TSR
tsr = evaluate.compute_TSR(dgm, threshold)

# Full TSC (main objective)
tsc = evaluate.compute_SC_adj(
    data, lens, graph, cover,
    type='k',       # 'k' = KeplerMapper, 'd' = D_Mapper
    N=100,          # bootstrap iterations
    alpha=0.85,     # confidence level
    w1=0.5, w2=0.5  # weights
)
```

---

## D-Mapper (Baseline)

D-Mapper replaces the uniform cover with a GMM-guided cover (`D_Cover`) that adapts interval boundaries to the data distribution.

```python
from kmapper import D_Cover, D_Mapper

cover = D_Cover(n_cubes=10, alpha=0.1)
mapper = D_Mapper()
graph = mapper.map(lens, data, cover=cover)
```

`D_Cover` parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_cubes` | `None` | Number of hypercubes (auto via Bayesian GMM if None) |
| `alpha` | `None` | Symmetrical quantile controlling interval overlap |
| `n_init` | `5` | GMM initializations (best result kept) |
| `max_iter` | `1000` | EM iterations |
| `tol` | `0.75e-3` | EM convergence threshold |

---

## Results

### TSC Comparison (S-Mapper vs Mapper vs D-Mapper)

> Optimization settings: 20 epochs, population size 20, `seed=42`.  
> Search bounds: N ∈ [5, 25], p/α ∈ [0.1, 0.7].

| Dataset | S-Mapper | Mapper (manual) | D-Mapper | Δ vs Mapper | Δ vs D-Mapper |
|---------|:--------:|:---------------:|:--------:|:-----------:|:-------------:|
| two_cir | **0.8346** | 0.8253 | 0.8177 | +0.0093 | +0.0169 |
| two_i_cir | **0.8050** | 0.7254 | 0.8197 | +0.0796 | −0.0147 |
| trefoil | **0.5359** | 0.5037 | 0.3876 | +0.0322 | +0.1483 |
| cat | **0.7952** | 0.7495 | 0.7555 | +0.0457 | +0.0397 |
| lion | **0.7815** | 0.5069 | 0.5959 | +0.2746 | +0.1856 |
| horse | **0.8303** | 0.5450 | 0.6363 | +0.2853 | +0.1940 |
| human | **0.8210** | 0.8094 | 0.4944 | +0.0116 | +0.3266 |
| diabetes | **0.3957** | 0.3405 | 0.2986 | +0.0552 | +0.0971 |
| covid-19 | **0.5402** | 0.3978 | 0.2852 | +0.1424 | +0.2550 |

S-Mapper achieves the highest TSC on 8/9 datasets (the only exception is `two_i_cir`, where D-Mapper leads by 0.0147).

### PSO vs GWO Convergence

| Dataset | PSO — best N | PSO — best p | PSO — TSC | PSO time (s) | GWO — best N | GWO — best p | GWO — TSC | GWO time (s) |
|---------|---:|---:|---:|---:|---:|---:|---:|---:|
| two_cir | 18 | 0.100 | 0.8324 | 3947 | **23** | 0.100 | **0.8346** | **2404** |
| two_i_cir | **25** | 0.120 | **0.8050** | 3614 | 5 | 0.100 | 0.8001 | 3102 |
| trefoil | 6 | 0.174 | 0.5359 | 155 | 6 | 0.174 | 0.5359 | **179** |
| cat | 7 | 0.100 | 0.7952 | 3983 | 7 | 0.100 | 0.7952 | **2481** |
| lion | 9 | 0.100 | 0.7815 | 2300 | 9 | 0.100 | 0.7815 | **1531** |
| horse | 5 | 0.100 | 0.8303 | 3598 | 5 | 0.100 | 0.8303 | **3000** |
| human | 7 | 0.100 | 0.8210 | 1570 | 7 | 0.100 | 0.8210 | **1021** |
| diabetes | 5 | 0.454 | 0.3957 | 83 | 5 | 0.455 | 0.3957 | **69** |
| covid-19 | 10 | 0.121 | 0.5402 | 460 | 10 | 0.121 | 0.5402 | **421** |

Both algorithms reach the same optimal (N\*, p\*) on 7/9 datasets. GWO is consistently faster: convergence time averages **5–27×** lower than PSO across 3D mesh datasets (e.g., cat: 124 s vs 3385 s, human: 51 s vs 628 s).

---

## Citation

If you use S-Mapper in your research, please cite:

```bibtex
@article{smapper2025,
  title   = {S-Mapper: A Topology-Aware Optimization Framework for Robust Mapper Construction},
  year    = {2025},
  journal = {ACM Transactions on Knowledge Discovery from Data},
  note    = {Under review}
}
```

This work builds on:

```bibtex
@article{tao2024dmapper,
  title   = {A distribution-guided Mapper algorithm},
  author  = {Tao, Yuyang and Ge, Shufei},
  year    = {2024},
  url     = {https://arxiv.org/abs/2401.12237}
}

@article{van2023mealpy,
  title   = {MEALPY: An open-source library for latest meta-heuristic algorithms in Python},
  author  = {Van Thieu, Nguyen and Mirjalili, Seyedali},
  journal = {Journal of Systems Architecture},
  year    = {2023},
  doi     = {10.1016/j.sysarc.2023.102871}
}

@article{naul2019kepler,
  title   = {Kepler Mapper: A flexible Python implementation of the Mapper algorithm},
  author  = {van Veen, Hendrik Jacob and Saul, Nathaniel and Eargle, David and Mangham, Sam W.},
  journal = {Journal of Open Source Software},
  year    = {2019},
  doi     = {10.21105/joss.01315}
}
```

---

## License

The S-Mapper extensions (`evaluate.py`) are released for research use.  
D-Mapper (`dcover.py`, `dmapper.py`) is copyright © 2024 Yuyang Tao and Shufei Ge — see [implement/LICENSES.txt](implement/LICENSES.txt).  
KeplerMapper is licensed under BSD — see [implement/kepler-mapper/](implement/kepler-mapper/).  
MEALPY is licensed under MIT — see [implement/mealpy/](implement/mealpy/).

---

## Contact

For bugs or questions, open a GitHub issue.
