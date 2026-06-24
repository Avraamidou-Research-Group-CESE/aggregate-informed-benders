using JLD2, Printf
import JuMP

include("electrolyzer_struct.jl")
include("formulation.jl")

# ── Stored solution ────────────────────────────────────────────────────────────

struct MonolithicSolution
    obj_value  ::Float64          # model objective: min(−NPV)  [incumbent]
    obj_bound  ::Float64          # best bound reported by Gurobi
    mip_gap    ::Float64          # |incumbent − bound| / |incumbent|
    npv        ::Float64          # total NPV = −obj_value  [excludes plant CAPEX]
    lcoh       ::Float64          # levelized cost of hydrogen [$/kg]
    z_replace  ::Vector{Float64}  # [n_years]   replacement decisions
    A          ::Vector{Float64}  # [T_span]    hourly efficiency
    z_on       ::Vector{Float64}  # [T_span]    on-state dispatch
    z_sb       ::Vector{Float64}  # [T_span]    standby dispatch
    h          ::Vector{Float64}  # [T_span]    hydrogen production [kg]
    n_years    ::Int
    gap        ::Float64          # MIPGap tolerance passed to solver
    solve_time ::Float64          # wall-clock seconds
end

# ── Cache path ────────────────────────────────────────────────────────────────

const _CACHE_DIR  = joinpath(@__DIR__, "..", "cache")
const _CACHE_FILE = joinpath(_CACHE_DIR, "monolithic_cache.jld2")

# ── Cache key ─────────────────────────────────────────────────────────────────
#
# Key = hash(constructor params + gap + z_fixed) ++ hash(price array).
# Constructor params fully determine all derived Electrolyzer fields, so only
# the 11 inputs to Electrolyzer(…) need to be included.

function _cache_key(θ::Electrolyzer, spp_array::AbstractVector{<:Real},
                    gap::Float64, z_fixed=nothing)::String
    params = (θ.ϕ, θ.ξ_start, θ.η_overpotential, θ.V_i, θ.ℓ,
              θ.η_startup, θ.λ_H, θ.i, θ.λ_CAPEX_Plant, θ.λ_CAPEX_Stack, θ.ρ_sb,
              gap, hash(z_fixed))
    string(hash(params), base=16) * "_" * string(hash(spp_array), base=16)
end

# ── JLD2 field-by-field I/O ───────────────────────────────────────────────────
#
# Fields are written individually under "key/field_name" paths rather than as
# a typed struct. This makes the cache robust to struct changes: new fields
# added later simply fall back to defaults when reading older entries.

function _read_entry(f::JLD2.JLDFile, key::String)::MonolithicSolution
    _get(field, default) = haskey(f, "$key/$field") ? f["$key/$field"] : default
    MonolithicSolution(
        f["$key/obj_value"],
        _get("obj_bound", f["$key/obj_value"]),  # default: bound == incumbent
        _get("mip_gap",   0.0),
        f["$key/npv"],
        f["$key/lcoh"],
        f["$key/z_replace"],
        f["$key/A"],
        f["$key/z_on"],
        f["$key/z_sb"],
        f["$key/h"],
        f["$key/n_years"],
        f["$key/gap"],
        f["$key/solve_time"],
    )
end

function _write_entry(f::JLD2.JLDFile, key::String, sol::MonolithicSolution)
    haskey(f, key) && delete!(f, key)
    f["$key/obj_value"]  = sol.obj_value
    f["$key/obj_bound"]  = sol.obj_bound
    f["$key/mip_gap"]    = sol.mip_gap
    f["$key/npv"]        = sol.npv
    f["$key/lcoh"]       = sol.lcoh
    f["$key/z_replace"]  = sol.z_replace
    f["$key/A"]          = sol.A
    f["$key/z_on"]       = sol.z_on
    f["$key/z_sb"]       = sol.z_sb
    f["$key/h"]          = sol.h
    f["$key/n_years"]    = sol.n_years
    f["$key/gap"]        = sol.gap
    f["$key/solve_time"] = sol.solve_time
end

# ── Public API ────────────────────────────────────────────────────────────────

"""
    cache_solution!(model, θ, spp_array; gap, z_fixed, solve_time) → MonolithicSolution

Persist a solved `run_3st_opt` model to the on-disk JLD2 cache and return the
stored `MonolithicSolution`. Call immediately after `optimize!(model)`.

    t = @elapsed model = run_3st_opt(θ, prices)
    sol = cache_solution!(model, θ, prices; gap=0.01, solve_time=t)
"""
function cache_solution!(model::JuMP.Model, θ::Electrolyzer,
                         spp_array::AbstractVector{<:Real};
                         gap::Float64        = 0.01,
                         z_fixed             = nothing,
                         solve_time::Float64 = 0.0)::MonolithicSolution

    key   = _cache_key(θ, Float64.(spp_array), gap, z_fixed)
    inc   = JuMP.objective_value(model)
    bound = JuMP.objective_bound(model)

    sol = MonolithicSolution(
        inc,
        bound,
        abs(inc - bound) / max(abs(inc), 1.0),
        -inc,
        JuMP.value(model[:LCOH]),
        Float64.(JuMP.value.(model[:z_replace])),
        Float64.(JuMP.value.(model[:A])),
        Float64.(JuMP.value.(model[:z_on])),
        Float64.(JuMP.value.(model[:z_sb])),
        Float64.(JuMP.value.(model[:h])),
        Int(ceil(length(spp_array) / 8760)),
        gap,
        solve_time,
    )

    mkpath(_CACHE_DIR)
    jldopen(_CACHE_FILE, "a+") do f
        _write_entry(f, key, sol)
    end

    @printf("Cached %d-year solution  LCOH = %.4f \$/kg  gap = %.4f%%  (key %.12s…)\n",
            sol.n_years, sol.lcoh, sol.mip_gap * 100, key)
    return sol
end

"""
    load_cached_solution(θ, spp_array; gap, z_fixed) → MonolithicSolution | nothing

Return the cached solution for this instance, or `nothing` on a miss.
"""
function load_cached_solution(θ::Electrolyzer, spp_array::AbstractVector{<:Real};
                               gap::Float64 = 0.01,
                               z_fixed      = nothing)::Union{MonolithicSolution, Nothing}
    key = _cache_key(θ, Float64.(spp_array), gap, z_fixed)
    isfile(_CACHE_FILE) || (@printf("Cache miss\n"); return nothing)
    jldopen(_CACHE_FILE, "r") do f
        haskey(f, key) || (@printf("Cache miss\n"); return nothing)
        sol = _read_entry(f, key)
        @printf("Cache hit — %d-year solution  LCOH = %.4f \$/kg  gap = %.4f%%  solve was %.0fs\n",
                sol.n_years, sol.lcoh, sol.mip_gap * 100, sol.solve_time)
        return sol
    end
end

"""
    run_3st_opt_cached(θ, spp_array; gap, z_fixed) → MonolithicSolution

Drop-in wrapper for `run_3st_opt`: returns the cached solution if available,
otherwise solves, caches, and returns a `MonolithicSolution`.
"""
function run_3st_opt_cached(θ::Electrolyzer, spp_array;
                             gap::Float64 = 0.01,
                             z_fixed      = nothing)::MonolithicSolution
    cached = load_cached_solution(θ, Float64.(spp_array); gap=gap, z_fixed=z_fixed)


    cached !== nothing && return cached
    t_solve = @elapsed model = run_3st_opt(θ, spp_array; gap=gap, z_fixed=z_fixed)
    return cache_solution!(model, θ, Float64.(spp_array);
                           gap=gap, z_fixed=z_fixed, solve_time=t_solve)
end


function run_3st_opt_cached_rtm(θ::Electrolyzer, spp_array;
                             gap::Float64 = 0.01,
                             z_fixed      = nothing)::MonolithicSolution

                             cached = load_cached_solution(θ, Float64.(spp_array); gap=gap, z_fixed=z_fixed)
    cached !== nothing && return cached
    t_solve = @elapsed model = run_3st_opt_rtm(θ, spp_array; gap=gap, z_fixed = z_fixed )
    return cache_solution!(model, θ, Float64.(spp_array);
                           gap=gap, z_fixed=z_fixed, solve_time=t_solve)
end
"""
    list_cache() → nothing

Print all entries in the cache with their metadata.
"""
function list_cache()
    isfile(_CACHE_FILE) || (@printf("Cache is empty.\n"); return)
    jldopen(_CACHE_FILE, "r") do f
        top_keys = unique(first.(split.(keys(f), "/")))
        isempty(top_keys) && (@printf("Cache is empty.\n"); return)
        entries = [(k, _read_entry(f, String(k))) for k in top_keys]
        @printf("%-28s  %7s  %10s  %14s  %14s  %10s  %10s\n",
                "Key", "n_years", "LCOH (\$/kg)", "incumbent", "best bound", "gap (%)", "solve (s)")
        @printf("%s\n", "─"^100)
        for (k, s) in sort(entries; by = x -> x[2].n_years)
            @printf("%.26s…  %7d  %10.4f  %14.1f  %14.1f  %10.4f  %10.0f\n",
                    k, s.n_years, s.lcoh, s.obj_value, s.obj_bound, s.mip_gap * 100, s.solve_time)
        end
    end
end

"""
    delete_cached_solution!(θ, spp_array; gap, z_fixed) → Bool

Remove a specific entry from the cache. Returns `true` if the entry existed.
"""
function delete_cached_solution!(θ::Electrolyzer, spp_array::AbstractVector{<:Real};
                                  gap::Float64 = 0.01, z_fixed = nothing)::Bool
    key = _cache_key(θ, Float64.(spp_array), gap, z_fixed)
    isfile(_CACHE_FILE) || (@printf("Entry not found.\n"); return false)
    existed = false
    jldopen(_CACHE_FILE, "a+") do f
        if haskey(f, key)
            delete!(f, key)
            existed = true
        end
    end
    @printf("%s\n", existed ? "Entry deleted." : "Entry not found.")
    return existed
end
