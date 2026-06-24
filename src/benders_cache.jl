using JLD2, Printf

# ── Stored solution ────────────────────────────────────────────────────────────
#
# yearly_subprobs and cut vectors are JuMP/struct objects that can't be
# round-tripped through JLD2 reliably, so only the scalar convergence metrics
# and the replacement schedule are persisted.

struct BendersSolution
    method      ::String           # "aggregate" | "k_aggregate"
    n_years     ::Int
    k_clusters  ::Int              # 0 for standard aggregate_benders
    mkt_type    ::String           # "DAM" | "RTM"
    tol         ::Float64          # convergence tolerance ϵ
    k_max       ::Int              # iteration cap passed to the solver
    LB_final    ::Float64
    UB_final    ::Float64
    gap_final   ::Float64          # |UB - LB| / |LB| at termination
    LBs         ::Vector{Float64}  # lower-bound history (one entry per iteration)
    UBs         ::Vector{Float64}  # upper-bound history
    n_iters     ::Int              # iterations at termination
    iter_times  ::Vector{Float64}  # wall-clock seconds per iteration
    total_time  ::Float64          # sum(iter_times)
    z_rep_final ::Vector{Int}      # [n_years] replacement schedule
end

# ── Cache paths ───────────────────────────────────────────────────────────────

# const _BENDERS_CACHE_DIR  = joinpath(@__DIR__, "..", "cache")
const _BENDERS_CACHE_DIR  = joinpath(@__DIR__, "..", "bernard_cache")
const _BENDERS_CACHE_FILE = joinpath(_BENDERS_CACHE_DIR, "benders_cache.jld2")

# Re-export the monolithic cache file path so wipe_cache! can clear both.
# (Defined in solution_cache.jl as _CACHE_FILE; we reference by path here to
# avoid load-order coupling.)
const _MONO_CACHE_FILE = joinpath(_BENDERS_CACHE_DIR, "monolithic_cache.jld2")

# ── Cache key ─────────────────────────────────────────────────────────────────

function _benders_cache_key(θ::Electrolyzer, n_years::Int, method::String,
                             k_clusters::Int, mkt_type::String, tol::Float64,
                             k_max::Int)::String
    params = (θ.ϕ, θ.ξ_start, θ.η_overpotential, θ.V_i, θ.ℓ,
              θ.η_startup, θ.λ_H, θ.i, θ.λ_CAPEX_Plant, θ.λ_CAPEX_Stack, θ.ρ_sb,
              n_years, method, k_clusters, mkt_type, tol, k_max)
    string(hash(params), base=16)
end

# ── JLD2 field-by-field I/O ───────────────────────────────────────────────────

function _read_benders_entry(f::JLD2.JLDFile, key::String)::BendersSolution
    _get(field, default) = haskey(f, "$key/$field") ? f["$key/$field"] : default
    BendersSolution(
        f["$key/method"],
        f["$key/n_years"],
        _get("k_clusters", 0),
        f["$key/mkt_type"],
        f["$key/tol"],
        _get("k_max", 30),
        f["$key/LB_final"],
        f["$key/UB_final"],
        f["$key/gap_final"],
        Float64.(f["$key/LBs"]),
        Float64.(f["$key/UBs"]),
        f["$key/n_iters"],
        Float64.(f["$key/iter_times"]),
        f["$key/total_time"],
        Int.(f["$key/z_rep_final"]),
    )
end

function _write_benders_entry(f::JLD2.JLDFile, key::String, sol::BendersSolution)
    haskey(f, key) && delete!(f, key)
    f["$key/method"]      = sol.method
    f["$key/n_years"]     = sol.n_years
    f["$key/k_clusters"]  = sol.k_clusters
    f["$key/mkt_type"]    = sol.mkt_type
    f["$key/tol"]         = sol.tol
    f["$key/k_max"]       = sol.k_max
    f["$key/LB_final"]    = sol.LB_final
    f["$key/UB_final"]    = sol.UB_final
    f["$key/gap_final"]   = sol.gap_final
    f["$key/LBs"]         = sol.LBs
    f["$key/UBs"]         = sol.UBs
    f["$key/n_iters"]     = sol.n_iters
    f["$key/iter_times"]  = sol.iter_times
    f["$key/total_time"]  = sol.total_time
    f["$key/z_rep_final"] = sol.z_rep_final
end

# ── Internal helper: pack Benders return tuple → BendersSolution ──────────────

function _pack_benders(LBs_raw, UBs_raw, k::Int, iter_times_raw, z_rep_final_raw,
                       method::String, n_years::Int, k_clusters::Int,
                       mkt_type::String, tol::Float64, k_max::Int)::BendersSolution
    LBs_f  = Float64.(collect(LBs_raw))
    UBs_f  = Float64.(collect(UBs_raw))
    its_f  = Float64.(iter_times_raw)
    LB     = LBs_f[end]
    UB     = UBs_f[end]
    gap    = abs(UB - LB) / max(abs(LB), 1.0)
    BendersSolution(method, n_years, k_clusters, mkt_type, tol, k_max,
                    LB, UB, gap, LBs_f, UBs_f, k, its_f, sum(its_f),
                    Int.(z_rep_final_raw))
end

# ── Public API ────────────────────────────────────────────────────────────────

"""
    cache_benders_solution!(θ, LBs, UBs, k, iter_times, z_rep_final;
                            method, n_years, k_clusters, mkt_type, tol) → BendersSolution

Build and persist a `BendersSolution` directly from the raw Benders return values.
"""
function cache_benders_solution!(θ::Electrolyzer,
                                  LBs_raw, UBs_raw, k::Int,
                                  iter_times_raw, z_rep_final_raw;
                                  method::String   = "aggregate",
                                  n_years::Int     = length(z_rep_final_raw),
                                  k_clusters::Int  = 0,
                                  mkt_type::String = "DAM",
                                  tol::Float64     = 0.01,
                                  k_max::Int       = 60)::BendersSolution
    sol = _pack_benders(LBs_raw, UBs_raw, k, iter_times_raw, z_rep_final_raw,
                        method, n_years, k_clusters, mkt_type, tol, k_max)
    key = _benders_cache_key(θ, n_years, method, k_clusters, mkt_type, tol, k_max)
    mkpath(_BENDERS_CACHE_DIR)
    jldopen(_BENDERS_CACHE_FILE, "a+") do f
        _write_benders_entry(f, key, sol)
    end
    @printf("Cached %s  n_years=%d  k=%d  LB=%.1f  UB=%.1f  gap=%.4f%%  t=%.0fs  (key %.12s…)\n",
            sol.method, sol.n_years, sol.n_iters,
            sol.LB_final, sol.UB_final, sol.gap_final * 100, sol.total_time, key)
    return sol
end

"""
    load_cached_benders_solution(θ, n_years; method, k_clusters, mkt_type, tol)
        → BendersSolution | nothing

Return the cached solution for this instance, or `nothing` on a miss.
"""
function load_cached_benders_solution(θ::Electrolyzer, n_years::Int;
                                       method::String   = "aggregate",
                                       k_clusters::Int  = 0,
                                       mkt_type::String = "DAM",
                                       tol::Float64     = 0.01,
                                       k_max::Int       = 60
                                       )::Union{BendersSolution, Nothing}
    key = _benders_cache_key(θ, n_years, method, k_clusters, mkt_type, tol, k_max)
    isfile(_BENDERS_CACHE_FILE) || (@printf("Benders cache miss\n"); return nothing)
    jldopen(_BENDERS_CACHE_FILE, "r") do f
        haskey(f, key) || (@printf("Benders cache miss\n"); return nothing)
        sol = _read_benders_entry(f, key)
        @printf("Benders cache hit — %s  n_years=%d  k=%d  LB=%.1f  UB=%.1f  gap=%.4f%%  t=%.0fs\n",
                sol.method, sol.n_years, sol.n_iters,
                sol.LB_final, sol.UB_final, sol.gap_final * 100, sol.total_time)
        return sol
    end
end

"""
    traditional_benders_cached(θ, n_years; type, T_length, ϵ) → BendersSolution

Drop-in cached wrapper for `traditional_benders`. Returns a `BendersSolution` with
the convergence history and replacement schedule. On a cache hit the solver is
skipped entirely; on a miss the result is solved, cached, and returned.

The full Benders return tuple (yearly_subprobs, cuts, …) is not serialisable, so
callers that need the JuMP models must call `traditional_benders` directly.
"""
function traditional_benders_cached(θ::Electrolyzer, n_years::Int;
                                     type::String  = "DAM",
                                     T_length::Int = 8760,
                                     ϵ::Float64    = 0.01,
                                     k_max::Int    = 60)::BendersSolution
    cached = load_cached_benders_solution(θ, n_years;
                                          method="traditional", k_clusters=0,
                                          mkt_type=type, tol=ϵ, k_max=k_max)
    cached !== nothing && return cached

    _, _, _, LBs, UBs, k, iter_times, z_rep_final =
        traditional_benders(θ, n_years; type=type, T_length=T_length, ϵ=ϵ, k_max=k_max)

    return cache_benders_solution!(θ, LBs, UBs, k, iter_times, z_rep_final;
                                   method="traditional", n_years=n_years,
                                   k_clusters=0, mkt_type=type, tol=ϵ, k_max=k_max)
end

"""
    aggregate_benders_cached(θ, n_years; type, T_length, ϵ) → BendersSolution

Drop-in cached wrapper for `aggregate_benders`. Returns a `BendersSolution` with
the convergence history and replacement schedule. On a cache hit the solver is
skipped entirely; on a miss the result is solved, cached, and returned.

The full Benders return tuple (yearly_subprobs, cuts, …) is not serialisable, so
callers that need the JuMP models must call `aggregate_benders` directly.
"""
function aggregate_benders_cached(θ::Electrolyzer, n_years::Int;
                                   type::String  = "DAM",
                                   T_length::Int = 8760,
                                   ϵ::Float64    = 0.01,
                                   k_max::Int    = 60)::BendersSolution
    cached = load_cached_benders_solution(θ, n_years;
                                          method="aggregate", k_clusters=0,
                                          mkt_type=type, tol=ϵ, k_max=k_max)
    cached !== nothing && return cached

    _, _, _, LBs, UBs, k, iter_times, z_rep_final =
        aggregate_benders(θ, n_years; type=type, T_length=T_length, ϵ=ϵ, k_max=k_max)

    return cache_benders_solution!(θ, LBs, UBs, k, iter_times, z_rep_final;
                                   method="aggregate", n_years=n_years,
                                   k_clusters=0, mkt_type=type, tol=ϵ, k_max=k_max)
end

"""
    aggregate_k_benders_cached(θ, n_years, k_clusters; type, T_length, ϵ) → BendersSolution

Drop-in cached wrapper for `aggregate_k_benders`.
"""
function aggregate_k_benders_cached(θ::Electrolyzer, n_years::Int, k_clusters::Int;
                                     type::String  = "DAM",
                                     T_length::Int = 8760,
                                     ϵ::Float64    = 0.01,
                                     k_max::Int    = 60)::BendersSolution
    cached = load_cached_benders_solution(θ, n_years;
                                          method="k_aggregate", k_clusters=k_clusters,
                                          mkt_type=type, tol=ϵ, k_max=k_max)
    cached !== nothing && return cached

    _, _, _, LBs, UBs, k, iter_times, z_rep_final =
        aggregate_k_benders(θ, n_years, k_clusters; type=type, T_length=T_length, ϵ=ϵ, k_max=k_max)

    return cache_benders_solution!(θ, LBs, UBs, k, iter_times, z_rep_final;
                                   method="k_aggregate", n_years=n_years,
                                   k_clusters=k_clusters, mkt_type=type, tol=ϵ, k_max=k_max)
end

"""
    list_benders_cache() → nothing

Print all entries in the Benders cache with their metadata.
"""
function list_benders_cache()
    isfile(_BENDERS_CACHE_FILE) || (@printf("Benders cache is empty.\n"); return)
    jldopen(_BENDERS_CACHE_FILE, "r") do f
        top_keys = unique(first.(split.(keys(f), "/")))
        isempty(top_keys) && (@printf("Benders cache is empty.\n"); return)
        entries = [(k, _read_benders_entry(f, String(k))) for k in top_keys]
        @printf("%-28s  %-12s  %4s  %7s  %5s  %6s  %5s  %5s  %14s  %14s  %9s  %10s\n",
                "Key", "method", "mkt", "n_years", "k_cl", "tol", "k_max", "iters", "LB", "UB", "gap (%)", "time (s)")
        @printf("%s\n", "─"^130)
        for (k, s) in sort(entries; by = x -> (x[2].method, x[2].n_years, x[2].k_clusters))
            @printf("%.26s…  %-12s  %4s  %7d  %5d  %6.4f  %5d  %5d  %14.1f  %14.1f  %9.4f  %10.0f\n",
                    k, s.method, s.mkt_type, s.n_years, s.k_clusters, s.tol, s.k_max, s.n_iters,
                    s.LB_final, s.UB_final, s.gap_final * 100, s.total_time)
        end
    end
end

"""
    inspect_benders_cache_entry(θ, n_years; method, k_clusters, mkt_type, tol, k_max) → nothing

Print a detailed view of a single cache entry, including all convergence history.
"""
function inspect_benders_cache_entry(θ::Electrolyzer, n_years::Int;
                                      method::String   = "aggregate",
                                      k_clusters::Int  = 0,
                                      mkt_type::String = "DAM",
                                      tol::Float64     = 0.01,
                                      k_max::Int       = 60)
    sol = load_cached_benders_solution(θ, n_years;
                                       method=method, k_clusters=k_clusters,
                                       mkt_type=mkt_type, tol=tol, k_max=k_max)
    sol === nothing && return

    @printf("\n══════════════════════════════════════════════════════\n")
    @printf("  Benders Cache Entry\n")
    @printf("══════════════════════════════════════════════════════\n")
    @printf("  method      : %s\n",  sol.method)
    @printf("  mkt_type    : %s\n",  sol.mkt_type)
    @printf("  n_years     : %d\n",  sol.n_years)
    @printf("  k_clusters  : %d\n",  sol.k_clusters)
    @printf("  tol (ϵ)     : %.4f\n", sol.tol)
    @printf("  k_max       : %d\n",  sol.k_max)
    @printf("──────────────────────────────────────────────────────\n")
    @printf("  LB_final    : %.4f\n", sol.LB_final)
    @printf("  UB_final    : %.4f\n", sol.UB_final)
    @printf("  gap         : %.4f%%\n", sol.gap_final * 100)
    @printf("  n_iters     : %d\n",  sol.n_iters)
    @printf("  total_time  : %.1f s\n", sol.total_time)
    @printf("──────────────────────────────────────────────────────\n")
    @printf("  z_rep       : %s\n",  string(sol.z_rep_final))
    @printf("──────────────────────────────────────────────────────\n")
    @printf("  iter   LB              UB              gap (%%)    time (s)\n")
    @printf("  %s\n", "─"^62)
    for i in eachindex(sol.LBs)
        gap_i = abs(sol.UBs[i] - sol.LBs[i]) / max(abs(sol.LBs[i]), 1.0) * 100
        @printf("  %4d  %14.1f  %14.1f  %9.4f  %10.2f\n",
                i, sol.LBs[i], sol.UBs[i], gap_i, sol.iter_times[i])
    end
    @printf("══════════════════════════════════════════════════════\n\n")
    return nothing
end

# ── Single-entry deletion ─────────────────────────────────────────────────────

"""
    delete_cached_benders_solution!(θ, n_years; method, k_clusters, mkt_type, tol, k_max) → Bool

Remove a specific entry from the Benders cache. Keyword arguments must match
exactly what was passed when the entry was created. Returns `true` if the entry
existed and was deleted, `false` if it was not found.

Example — remove a bad traditional-Benders run for 20-year DAM at k_max=60:

    delete_cached_benders_solution!(θ, 20;
        method="traditional", k_clusters=0, mkt_type="DAM", tol=0.01, k_max=60)
"""
function delete_cached_benders_solution!(θ::Electrolyzer, n_years::Int;
                                          method::String   = "aggregate",
                                          k_clusters::Int  = 0,
                                          mkt_type::String = "DAM",
                                          tol::Float64     = 0.01,
                                          k_max::Int       = 60)::Bool
    key = _benders_cache_key(θ, n_years, method, k_clusters, mkt_type, tol, k_max)
    isfile(_BENDERS_CACHE_FILE) || (@printf("Entry not found.\n"); return false)
    existed = false
    jldopen(_BENDERS_CACHE_FILE, "a+") do f
        if haskey(f, key)
            delete!(f, key)
            existed = true
        end
    end
    @printf("%s\n", existed ? "Entry deleted." : "Entry not found.")
    return existed
end

# ── Cache wipe ────────────────────────────────────────────────────────────────

"""
    wipe_cache!() → nothing

Delete all cache files (Benders and monolithic). Irreversible.
"""
function wipe_cache!()
    wiped = String[]
    for path in (_BENDERS_CACHE_FILE, _MONO_CACHE_FILE)
        if isfile(path)
            rm(path)
            push!(wiped, basename(path))
        end
    end
    if isempty(wiped)
        @printf("Cache already empty — nothing to wipe.\n")
    else
        @printf("Wiped: %s\n", join(wiped, ", "))
    end
    return nothing
end
