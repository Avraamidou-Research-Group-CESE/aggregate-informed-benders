using Printf

include("src/benders.jl")
include("src/solution_cache.jl")   # includes formulation.jl internally
include("src/benders_cache.jl")

# ── Sweep configuration ───────────────────────────────────────────────────────

const HORIZONS    = [10, 20, 40]
const MKT_TYPES   = ["DAM", "RTM"]
const K_CLUSTERS  = [12, 24]        # k=1 (aggregate) is always included
const BENDERS_ε   = 0.01
const ITER_MAX    = 60

θ = Electrolyzer(
    2.2,              # ϕ  — nominal capacity [MW]
    0.65,             # ξ_start — new-stack efficiency [% LHV]
    3.2 / 1_000_000,  # η_overpotential [μV/hr]
    1.9,              # V_i — cell voltage [V]
    10,               # ℓ  — stack lifetime [years]
    0.00204 / 500,    # η_startup [μV/hr]
    3,                # λ_H — hydrogen price [$/kg]
    0.05,             # i  — discount rate
    1816 * 1_000,     # λ_CAPEX_Plant [$/MW]
    250  * 1_000,     # λ_CAPEX_Stack [$/MW]
    0.05,             # ρ_sb — standby power ratio
)

# ── Result record ─────────────────────────────────────────────────────────────

struct SweepResult
    mkt_type   ::String
    n_years    ::Int
    method     ::String          # "traditional" | "k1" | "k12" | "k24" (etc.)
    k_clusters ::Int
    LB         ::Float64
    UB         ::Float64
    gap_pct    ::Float64
    n_iters    ::Int
    total_time ::Float64
end

results = SweepResult[]

# ── Helper: run one method and return a SweepResult ──────────────────────────

function run_traditional(θ, N, mkt, T_len)
    println("\n" * "="^65)
    @printf("\tTraditional Benders  (%d years, %s)\n", N, mkt)
    println("="^65)
    sol = traditional_benders_cached(θ, N;
              type=mkt, T_length=T_len, ϵ=BENDERS_ε, k_max=ITER_MAX)
    SweepResult(mkt, N, "traditional", 0,
                sol.LBs[end], sol.UBs[end], sol.gap_final * 100,
                sol.n_iters, sol.total_time)
end

function run_k1(θ, N, mkt, T_len)
    println("\n" * "="^65)
    @printf("\tAggregate k=1 Benders  (%d years, %s)\n", N, mkt)
    println("="^65)
    sol = aggregate_benders_cached(θ, N;
              type=mkt, T_length=T_len, ϵ=BENDERS_ε, k_max=ITER_MAX)
    SweepResult(mkt, N, "k1", 1,
                sol.LBs[end], sol.UBs[end], sol.gap_final * 100,
                sol.n_iters, sol.total_time)
end

function run_k(θ, N, mkt, T_len, k)
    println("\n" * "="^65)
    @printf("\tAggregate k=%d Benders  (%d years, %s)\n", k, N, mkt)
    println("="^65)
    sol = aggregate_k_benders_cached(θ, N, k;
              type=mkt, T_length=T_len, ϵ=BENDERS_ε, k_max=ITER_MAX)
    SweepResult(mkt, N, "k$k", k,
                sol.LBs[end], sol.UBs[end], sol.gap_final * 100,
                sol.n_iters, sol.total_time)
end

# ── Main sweep — DAM first, RTM second ───────────────────────────────────────

function run_combo!(results, θ, N, mkt)
    T_len = 8760   # utils.jl multiplies by n_per_hour=4 internally for RTM

    println("\n" * "#"^65)
    @printf("#  Market: %-5s   Horizon: %d years\n", mkt, N)
    println("#"^65)

    push!(results, run_traditional(θ, N, mkt, T_len))
    push!(results, run_k1(θ, N, mkt, T_len))
    for k in K_CLUSTERS
        push!(results, run_k(θ, N, mkt, T_len, k))
    end

    combo = filter(r -> r.mkt_type == mkt && r.n_years == N, results)
    println()
    println("─"^75)
    @printf("  Summary — %s  N=%d\n", mkt, N)
    println("─"^75)
    @printf("  %-16s  %14s  %14s  %8s  %7s  %10s\n",
            "Method", "LB", "UB", "gap (%)", "iters", "time (s)")
    println("  " * "─"^71)
    for r in combo
        @printf("  %-16s  %14.1f  %14.1f  %8.4f  %7d  %10.1f\n",
                r.method, r.LB, r.UB, r.gap_pct, r.n_iters, r.total_time)
    end
    println("─"^75)
end

println("\n" * "█"^65)
println("█  PHASE 1 — DAM")
println("█"^65)
for N in HORIZONS
    run_combo!(results, θ, N, "DAM")
end

println("\n" * "█"^65)
println("█  PHASE 2 — RTM")
println("█"^65)
for N in HORIZONS
    run_combo!(results, θ, N, "RTM")
end

# ── Final aggregated summary ──────────────────────────────────────────────────

println("\n\n" * "="^90)
println("  FULL SWEEP SUMMARY")
println("="^90)
@printf("  %-5s  %7s  %-16s  %14s  %14s  %8s  %7s  %10s\n",
        "Mkt", "N", "Method", "LB", "UB", "gap (%)", "iters", "time (s)")
println("  " * "─"^83)
for mkt in MKT_TYPES, N in HORIZONS
    combo = filter(r -> r.mkt_type == mkt && r.n_years == N, results)
    for r in combo
        @printf("  %-5s  %7d  %-16s  %14.1f  %14.1f  %8.4f  %7d  %10.1f\n",
                r.mkt_type, r.n_years, r.method,
                r.LB, r.UB, r.gap_pct, r.n_iters, r.total_time)
    end
    println("  " * "─"^83)
end
println("="^90)

# ── Write CSV for post-processing ─────────────────────────────────────────────

const RESULTS_DIR = joinpath(@__DIR__, "results")
mkpath(RESULTS_DIR)
const CSV_PATH = joinpath(RESULTS_DIR, "sweep.csv")

open(CSV_PATH, "w") do io
    println(io, "mkt_type,n_years,method,k_clusters,LB,UB,gap_pct,n_iters,total_time_s")
    for r in results
        @printf(io, "%s,%d,%s,%d,%.4f,%.4f,%.6f,%d,%.2f\n",
                r.mkt_type, r.n_years, r.method, r.k_clusters,
                r.LB, r.UB, r.gap_pct, r.n_iters, r.total_time)
    end
end
@printf("\nResults written to %s\n", CSV_PATH)
