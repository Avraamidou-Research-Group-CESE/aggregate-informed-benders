# Aggregation-Informed Benders Decomposition for Multi-Scale Optimization Framework

Code companion to the FOCAPO-CPC 2027 paper:
**"Decomposing a Multi-Scale Optimization Framework for Grid-Integrated Electrolysis using Aggregation-Informed Benders"**
Kiernan X. Jennings, Styliana Avraamidou, Victor M. Zavala — UW-Madison / Argonne National Laboratory

---

## Overview

This repository implements a Benders decomposition framework for optimizing the dispatch and stack-replacement schedule of an alkaline water electrolyzer over a multi-year planning horizon. The problem is formulated as a mixed-integer linear program (MILP) that minimizes net electricity costs subject to hydrogen demand, electrolyzer degradation dynamics, and NPV-based economics driven by ERCOT Day-Ahead Market (DAM) or Real-Time Market (RTM) prices.

The key computational challenge is that operation dispatch decisions are tightly coupled across years through the degradation state variable `A[t]`. Benders decomposition exploits this structure by separating the binary replacement decisions (master problem) from the annual dispatch (subproblems).

Three master-problem variants are implemented and benchmarked:

| Method | Master problem | Description |
|---|---|---|
| `traditional` | Binary `z_rep` only | No relaxation of subproblem structure in MP |
| `k=1` aggregate | Adds annual-aggregate operational variables | Single-cluster relaxation per year |
| `k`-aggregate | Adds `k`-cluster operational variables | k-clustered relaxation per year |

---

## Repository Structure

```
FINAL/
├── main.jl               # Run script
├── Project.toml          # Julia package environment
├── Manifest.toml         # Pinned dependency versions
├── data/
│   ├── ERCOT_DAM_AVG_2014-2024.xlsx   # Hourly DAM settlement point prices
│   └── ERCOT_15RTM_2014-2024.xlsx     # 15-min RTM settlement point prices
├── src/
│   ├── electrolyzer_struct.jl   # Electrolyzer device model and parameters
│   ├── utils.jl                 # Data loading, global param setup, helpers
│   ├── formulation.jl           # Monolithic 3-state MIP (JuMP)
│   ├── time_aggregate.jl        # Greedy time-series segmentation (k clusters)
│   ├── benders.jl               # Benders master and subproblem builders + solvers
│   ├── benders_cache.jl         # JLD2-backed cache for BendersSolution objects
│   └── solution_cache.jl        # JLD2-backed cache for MonolithicSolution objects
└── cache/                       # Auto-created on first run; stores .jld2 solve results
```

---

## Setup

**Prerequisites:** Julia ≥ 1.9, a valid Gurobi license (for MIP subproblems and master problems), and HiGHS (bundled via the Julia package).

```julia
# From the Julia REPL, activate the project and install dependencies
using Pkg
Pkg.activate("/path/to/FINAL")
Pkg.instantiate()
```

---

## Running the Code

### `main.jl`

This script runs all three Benders variants across multiple planning horizons and both market types, then writes a summary CSV.

```bash
julia --project=/path/to/FINAL main.jl
```

Or from the REPL:

```julia
julia> include("main.jl")
```

**What it does:**

1. Instantiates an `Electrolyzer` with the nominal parameters from the paper (2.2 MW, 65% LHV efficiency, 10-year stack lifetime, \$3/kg H₂ price, 5% discount rate).
2. Runs a nested sweep over:
   - **Horizons:** `N ∈ {10, 20, 40}` years
   - **Markets:** DAM (hourly), RTM (15-min)
   - **Methods:** traditional Benders, aggregate `k=1`, aggregate `k=12`, aggregate `k=24`
3. Convergence tolerance is `ε = 0.01` (1%) with a cap of 60 iterations per run.
4. Results are cached in `cache/benders_cache.jld2` — re-running the script will load cached solves rather than re-solving, so individual runs can be safely interrupted and resumed.
5. Prints a formatted summary table per `(market, horizon)` block and a final aggregated table.
6. Writes `results/sweep.csv` with columns: `mkt_type, n_years, method, k_clusters, LB, UB, gap_pct, n_iters, total_time_s`.

**Configurable constants at the top of `main.jl`:**

```julia
const HORIZONS   = [10, 20, 40]      # planning horizons to sweep
const MKT_TYPES  = ["DAM", "RTM"]    # markets to sweep
const K_CLUSTERS = [12, 24]          # k values for k-aggregate (k=1 always included)
const BENDERS_ε  = 0.01              # convergence tolerance
const ITER_MAX   = 60                # max Benders iterations
```

## Source Files

### `src/electrolyzer_struct.jl`
Defines the `Electrolyzer` struct. The constructor auto-computes `α_max` (maximum efficiency), `α_min`, degradation rates `δ_on` and `δ_start`, OPEX, and annualized CAPEX via the capital recovery factor (CRF).

### `src/formulation.jl`
Implements the full monolithic 3-state MIP in JuMP. See the paper for references to the framework.

### `src/benders.jl`
Core decomposition logic. Contains:
- **`build_masterproblem`** — traditional MP with `z_rep` and `θ[y]` cost-to-go variables.
- **`build_aggregate_masterproblem`** — augmented MP with single-cluster aggregate operational variables per year; provides a tighter LB.
- **`build_k_aggregate_masterproblem`** — augmented MP with `k` time clusters per year; clusters are computed by `greedy_time_series_segmentation` and represented by their minimum price.
- **`build_subproblem`** — full hourly MIP for a single year with `z_replace` and `A_init` fixed.
- **`traditional_benders`**, **`aggregate_benders`**, **`aggregate_k_benders`** — Benders loops with warm-start cuts, MIP subproblem caching (by `(y, z, A_init, x_on_prev, x_sb_prev)`), optimality cuts (LP dual on `A_prev`), simple integer L-shaped cuts, and no-good feasibility cuts.

### `src/time_aggregate.jl`
Implements a greedy top-down time-series segmentation algorithm. Given a price vector and target number of clusters `k`, it recursively splits the interval that maximally reduces the sum-of-squared-errors (SSE), using a sparse table for O(1) range-minimum queries. The representative price for each segment is its minimum (a conservative lower bound for the MP relaxation).

### `src/benders_cache.jl`
JLD2-backed persistence for `BendersSolution` structs. Cache keys are hashes of all electrolyzer parameters plus `(n_years, method, k_clusters, mkt_type, tol, k_max)`. Provides `traditional_benders_cached`, `aggregate_benders_cached`, `aggregate_k_benders_cached` as drop-in wrappers, plus `list_benders_cache()`, `delete_cached_benders_solution!()`, and `wipe_cache!()`.

### `src/solution_cache.jl`
Analogous JLD2 cache for the monolithic `MonolithicSolution`. Cache key additionally incorporates a hash of the price array, so different years or market types never collide.

---

## Output

**Console:** Iteration-level progress (`LB`, `UB`, gap %, cut counts, per-subproblem solve times) followed by a per-block summary table.

**`results/sweep.csv`:** One row per `(market, horizon, method)` with final bounds, gap, iteration count, and total wall time. Suitable for direct import into plotting scripts.

**`cache/`:** `.jld2` files storing solved instances. Delete this directory (or call `wipe_cache!()`) to force a full re-solve.

---

## Citation

If you use this code, please cite:

> K. X. Jennings, V. M. Zavala, S. Avraamidou, "Decomposing a Multi-Scale Optimization Framework for Grid-Integrated Electrolysis using Aggregation-Informed Benders," *FOCAPO-CPC 2027*.
