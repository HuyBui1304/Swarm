# S-Mapper: A Topology-Aware Optimization Framework for Robust Mapper Construction

> **Paper**: *S-Mapper: A Topology-Aware Optimization Framework for Robust Mapper Construction*  
> Submitted to *ACM Transactions on Knowledge Discovery from Data (TKDD)*, May 2025.

S-Mapper automatically selects the resolution parameters of the **Mapper** algorithm — number of intervals *N* and overlap percentage *p* — by maximizing a novel metric called the **Topological Silhouette Coefficient (TSC)** using Swarm Intelligence (PSO and GWO).

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

---

## Citation

```bibtex
@article{tao2025distribution,
  title={A distribution-guided Mapper algorithm},
  author={Tao, Yuyang and Ge, Shufei},
  journal={BMC Bioinformatics},
  volume={26},
  number={1},
  pages={73},
  year={2025},
  publisher={Springer}
}

@article{van2023mealpy,
  title={MEALPY: An open-source library for latest meta-heuristic algorithms in Python},
  author={Van Thieu, Nguyen and Mirjalili, Seyedali},
  journal={Journal of Systems Architecture},
  volume={139},
  pages={102871},
  year={2023},
  publisher={Elsevier}
}

@article{van2019kepler,
  title={Kepler Mapper: A flexible Python implementation of the Mapper algorithm.},
  author={Van Veen, Hendrik Jacob and Saul, Nathaniel and Eargle, David and Mangham, Sam W},
  journal={Journal of Open Source Software},
  volume={4},
  number={42},
  pages={1315},
  year={2019}
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

For bugs or questions, open a GitHub issue or email [huybm.ds@gmail.com](mailto:huybm.ds@gmail.com).
