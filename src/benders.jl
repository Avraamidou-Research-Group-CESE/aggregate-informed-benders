using JuMP, HiGHS, Gurobi, Printf, Suppressor
import MathOptInterface as MOI

include("utils.jl")
include("time_aggregate.jl")

struct OptimalityCut
    Q_LP::Float64
    π_dual::Float64
    A_prev::Float64
    z_cut::Int   # z_rep scenario this cut was generated under (0 or 1)
end

struct FeasibilityCut  # no-good cut: ∑_{j∈S1}(1-z[j]) + ∑_{j∈S0}z[j] >= 1
    z_k::Vector{Int}  # infeasible z_rep schedule
end

struct SimpleIntegerLCut # θ >= Q_MIP + (L_y - Q_MIP) * (1-z)
    Q_MIP::Float64 
    L_y::Float64 # valid lower bound on Q_MIP over the feasible range of a_init_y
    z_k::Int # if we replace in this year
    # y::Int # this is the year we are producing the cutting plane on
    # this can be used for indicator structure for tightening/loosening the cut structure
end

struct IntegerLCut # θ >= Q_MIP + (L_y - Q_MIP) * (1-z)
    Q_MIP::Float64
    L_y::Float64 # valid lower bound on Q_MIP over the feasible range of a_init_y
    z_k::Vector{Int} # if we replace in this year
    # y::Int # this is the year we are producing the cutting plane on
    # this can be used for indicator structure for tightening/loosening the cut structure
end

struct SubproblemCacheEntry
    Q_MIP::Float64
    A_terminal::Float64
    x_on_end::Float64
    x_sb_end::Float64
end

function build_masterproblem(elec::Electrolyzer, n_years::Int,
                    cuts::Vector{Vector{OptimalityCut}}, feas_cuts::Vector{FeasibilityCut}, 
                    sl_cuts::Vector{Vector{SimpleIntegerLCut}}, l_cuts::Vector{Vector{IntegerLCut}})
    """
    this is the basic mp with no aggregate information
    """
    Y  = 1:n_years
    mp = @suppress Model(Gurobi.Optimizer)

    @variables mp begin
        z_rep[Y], Bin
        θ[Y] >= -1e6
    end

    # cutting planes

    # integer feasibility cuts
    # ∑_{j∈S1}(1-z[j]) + ∑_{j∈S0}z[j] >= 1
    for cut in feas_cuts
        S1 = findall(==(1), cut.z_k)
        S0 = findall(==(0), cut.z_k)
        @constraint(mp,
            sum(1 - z_rep[j] for j in S1; init=0) +
            sum(z_rep[j]     for j in S0; init=0) >= 1)
    end

    # simple integer l-shaped cuts
    # active at z_k (θ >= Q_MIP), weakened to L_y for the opposite scenario
    for y in Y
        for cut in sl_cuts[y]
            if cut.z_k == 1
                @constraint(mp, θ[y] >= cut.Q_MIP + (cut.L_y - cut.Q_MIP) * (1 - z_rep[y]))
            # else
                # @constraint(mp, θ[y] >= cut.Q_MIP + (cut.L_y - cut.Q_MIP) * z_rep[y])
            end
        end
    end

    # l-shaped integer cuts
    # θ >= Q_MIP + (L_y - Q_MIP) * (Σ_{j∈S1} (1-z_j) + Σ_{j∈S0} (z_j))
    for y in Y
        for cut in l_cuts[y]
            S1 = findall(==(1), cut.z_k)
            S0 = findall(==(0), cut.z_k)
            @constraint(mp, θ[y] >= cut.Q_MIP + (cut.L_y - cut.Q_MIP) * (sum(1 - z_rep[j] for j in S1; init=0) + sum(z_rep[j] for j in S0; init=0)))
        end
    end

    @expression(mp, replace_cost[y in Y], df_vec[y] * elec.λ_CAPEX_replace * z_rep[y])
    @objective(mp, Min, sum(θ[y] for y in Y) + sum(replace_cost[y] for y in Y))

    return mp
end



function build_aggregate_masterproblem(elec::Electrolyzer, n_years::Int, lmp::Vector{Float64},
                    cuts::Vector{Vector{OptimalityCut}}, feas_cuts::Vector{FeasibilityCut}, 
                    sl_cuts::Vector{Vector{SimpleIntegerLCut}}, l_cuts::Vector{Vector{IntegerLCut}};
                    type = "DAM")
    """
    """
    if type == "DAM"
        T_y = 8760 # intervals in year
        D_y = 365 # days in year
        ΔT = 1
        γ = ΔT
    elseif type == "RTM"
        T_y = 8760*4 # intervals in year
        D_y = 365 # days in year
        ΔT = 0.25 # hrs / interval
        γ = ΔT
    end

    Y  = 1:n_years
    mp = @suppress Model(Gurobi.Optimizer)
    B = 9.66

    @variables mp begin
        z_rep[Y], Bin
        0 <= X_ON[Y] <= T_y
        0 <= X_OFF[Y] <= T_y
        0 <= X_SB[Y] <= T_y
        0 <= X_START[Y] <= T_y
        H[Y] >= 0
        E[Y] >= 0
        W[Y] >= 0
        V[Y] >= 0
        A[Y] >= elec.α_min
        θ[Y] >= -1e9
    end

    # operational modes
    @constraint(mp, [y in Y], X_ON[y] + X_OFF[y] + X_SB[y] == T_y)

    # startup constraints
    @constraint(mp, [y in Y], X_START[y] <=  X_ON[y])
    @constraint(mp, [y in Y], X_START[y] <=  T_y - X_ON[y])
    
    # A aggregate degradation expressions
    δ_on = elec.δ_on
    δ_start = elec.δ_start
    
    @constraint(mp, A[1] == elec.α_max - δ_on * γ * X_ON[1] - δ_start * X_START[1])
    @constraint(mp, [y in 2:n_years],
    A[y] == A[y-1] - δ_on * γ * X_ON[y] - δ_start * X_START[y] + elec.α_max * z_rep[y] - V[y])

    @constraint(mp, z_rep[1] == 0)

    # McCormick envelope: v[y] = A_agg[y-1] * z_rep[y]
    @constraint(mp, [y in Y], V[y] <= elec.α_max * z_rep[y])
    @constraint(mp, [y in Y], V[y] >= elec.α_min * z_rep[y])
    @constraint(mp, [y in 2:n_years], V[y] <= A[y-1] - elec.α_min * (1 - z_rep[y]))
    @constraint(mp, [y in 2:n_years], V[y] >= A[y-1] - elec.α_max * (1 - z_rep[y]))

    # hydrogen production
    @constraint(mp, [y in Y], H[y] == ϕ * γ * W[y] + B * γ * X_ON[y])

    @constraint(mp, [y in Y], W[y] <= elec.α_max * X_ON[y])
    @constraint(mp, [y in Y], W[y] >= elec.α_min * X_ON[y])
    @constraint(mp, [y in 2:n_years], W[y] <= A[y-1] * T_y + elec.α_max * T_y * z_rep[y])

    # @constraint(mp, [y in 2:n_years], W[y] <= A[y-1] + )

    # demand constraint
    σ = 750 # kg/day
    D_y = 365 # days in a year
    @constraint(mp, [y in Y], H[y] >= σ * D_y)

    # electricity consumption
    @constraint(mp, [y in Y], E[y] == ϕ * γ * X_ON[y] + ϕ * γ * elec.ρ_sb * X_SB[y])

    
    @expression(mp, df[y in Y], (1/(1+ρ))^(y-1))
    @expression(mp, h_revenue[y in Y], H[y] * elec.λ_H)
    @expression(mp, e_exp[y in Y], E[y] * lmp[y])
    # @expression(mp, npv_contrib, df[y]*(h_revenue - e_exp - λ_OPEX))
    @expression(mp, sp_obj[y in Y], df[y] * (e_exp[y] + elec.λ_OPEX - h_revenue[y]))

    # since this aggregate is a relaxation of the subproblem, this should be a valid LB on θ
    @constraint(mp, [y in Y], θ[y] >= sp_obj[y])

    # cutting planes
    # Benders optimality cuts
    M_cut = 1e9
    for y in Y
        A_start_expr = y == 1 ? elec.α_max : A[y-1]
        for cut in cuts[y]
            rhs = cut.Q_LP + cut.π_dual * (A_start_expr - cut.A_prev)
            if cut.z_cut == 0
                @constraint(mp, θ[y] >= rhs - M_cut * z_rep[y])
            else
                @constraint(mp, θ[y] >= rhs - M_cut * (1 - z_rep[y]))
            end
        end
    end

    # integer feasibility cuts
    # ∑_{j∈S1}(1-z[j]) + ∑_{j∈S0}z[j] >= 1
    for cut in feas_cuts
        S1 = findall(==(1), cut.z_k)
        S0 = findall(==(0), cut.z_k)
        @constraint(mp,
            sum(1 - z_rep[j] for j in S1; init=0) +
            sum(z_rep[j]     for j in S0; init=0) >= 1)
    end

    # simple integer l-shaped cuts
    # active at z_k (θ >= Q_MIP), weakened to L_y for the opposite scenario
    for y in Y
        for cut in sl_cuts[y]
            if cut.z_k == 1
                @constraint(mp, θ[y] >= cut.Q_MIP + (cut.L_y - cut.Q_MIP) * (1 - z_rep[y]))
            # else
                # @constraint(mp, θ[y] >= cut.Q_MIP + (cut.L_y - cut.Q_MIP) * z_rep[y])
            end
        end
    end

    # l-shaped integer cuts
    # θ >= Q_MIP + (L_y - Q_MIP) * (Σ_{j∈S1} (1-z_j) + Σ_{j∈S0} (z_j))
    for y in Y
        for cut in l_cuts[y]
            S1 = findall(==(1), cut.z_k)
            S0 = findall(==(0), cut.z_k)
            @constraint(mp, θ[y] >= cut.Q_MIP + (cut.L_y - cut.Q_MIP) * (sum(1 - z_rep[j] for j in S1; init=0) + sum(z_rep[j] for j in S0; init=0)))
        end
    end

    @expression(mp, replace_cost[y in Y], df_vec[y] * elec.λ_CAPEX_replace * z_rep[y])
    @objective(mp, Min, sum(θ[y] for y in Y) + sum(replace_cost[y] for y in Y))

    return mp
end



function build_k_aggregate_masterproblem(elec::Electrolyzer, n_years::Int, k_clusters::Int, lmp::Vector{Vector{Float64}},
                    cuts::Vector{Vector{OptimalityCut}}, feas_cuts::Vector{FeasibilityCut}, 
                    sl_cuts::Vector{Vector{SimpleIntegerLCut}}, l_cuts::Vector{Vector{IntegerLCut}};
                    type = "DAM")
    """
    extend the `build_aggregate_masterproblem` function to aggregate over the designated kth clusters
    """
    if type == "DAM"
        T_y = 8760 # intervals in year
        D_y = 365 # days in year
        ΔT = 1
        γ = ΔT
    elseif type == "RTM"
        T_y = 8760*4 # intervals in year
        D_y = 365 # days in year
        ΔT = 0.25 # hrs / interval
        γ = ΔT
    end

    K = 1:k_clusters
    Y  = 1:n_years
    mp = @suppress Model(Gurobi.Optimizer)
    B = 9.66
    T_k = [[T_y for _ in K] for _ in Y] # intervals in a given cluster of a given year, initialized to UB
    lmp_k = [[0.0 for _ in K] for _ in Y]
    # preprocessing step
    for y in Y
        intervals, min_prices = greedy_time_series_segmentation(lmp[y], k_clusters) # lmp[y] is the prices for the year
        # where intervals are the indices of the vector lmp
        # min_prices are the market prices for those indices


        # index T_k[y][k]
        for k_idx in K
            k_size = intervals[k_idx][2] - intervals[k_idx][1] + 1
            T_k[y][k_idx] = Float64.(k_size)
            lmp_k[y][k_idx] = min_prices[k_idx]
        end

    
    end


    @variables mp begin
        z_rep[Y], Bin
        V[Y] >= 0 # A[y-1] * z_rep[y]
        0 <= X_ON[Y,K] <= T_y
        0 <= X_OFF[Y,K] <= T_y
        0 <= X_SB[Y,K] <= T_y
        0 <= X_START[Y,K] <= T_y
        H[Y,K] >= 0
        E[Y,K] >= 0
        W[Y,K] >= 0
        A[Y,K] >= elec.α_min
        θ[Y] >= -1e9
    end
    # set a better UB on the variables
    @constraint(mp, [y in Y, k in K], X_ON[y, k] <= T_k[y][k]) 
    @constraint(mp, [y in Y, k in K], X_OFF[y, k] <= T_k[y][k]) 
    @constraint(mp, [y in Y, k in K], X_SB[y, k] <= T_k[y][k]) 
    @constraint(mp, [y in Y, k in K], X_START[y, k] <= T_k[y][k]) 

    # operational modes
    @constraint(mp, [y in Y, k in K], X_ON[y, k] + X_OFF[y, k] + X_SB[y, k] == T_k[y][k])

    # startup constraints
    @constraint(mp, [y in Y, k in K], X_START[y, k] <=  X_ON[y, k])
    @constraint(mp, [y in Y, k in K], X_START[y, k] <=  T_k[y][k] - X_ON[y, k])
    
    # A aggregate degradation expressions
    δ_on = elec.δ_on
    δ_start = elec.δ_start
    # initial condition
    @constraint(mp, A[1,1] == elec.α_max - δ_on * γ * X_ON[1, 1] - δ_start * X_START[1, 1])

    # replacement logic
    @constraint(mp, z_rep[1] == 0)

    # McCormick envelope: v[y] = A_agg[y-1] * z_rep[y]
    @constraint(mp, [y in Y], V[y] <= elec.α_max * z_rep[y])
    @constraint(mp, [y in Y], V[y] >= elec.α_min * z_rep[y])
    @constraint(mp, [y in 2:n_years], V[y] <= A[y-1, length(K)] - elec.α_min * (1 - z_rep[y]))
    @constraint(mp, [y in 2:n_years], V[y] >= A[y-1, length(K)] - elec.α_max * (1 - z_rep[y]))
    # replacement intervals
    for y in 2:length(Y)
        @constraint(mp, A[y, 1] == A[y-1, length(K)] - δ_on * γ * X_ON[y, 1] - δ_start * X_START[y, 1]
        + elec.α_max * z_rep[y] - V[y]) # including replacement
    end
    # intra years between clusters
    for y in Y
        if length(K) <= 2 error("k is too small for this current implementation"); break end
        # normal clusters
        for k in 2:length(K) 
            @constraint(mp, A[y, k] == A[y, k-1] - δ_on * γ * X_ON[y,k] - δ_start * X_START[y,k])
        end 
    end


    # hydrogen production
    @constraint(mp, [y in Y, k in K], H[y, k] == ϕ * γ * W[y, k] + B * γ * X_ON[y, k])

    @constraint(mp, [y in Y, k in K], W[y, k] <= elec.α_max * X_ON[y, k])
    @constraint(mp, [y in Y, k in K], W[y, k] >= elec.α_min * X_ON[y, k])
    # enforce production is constrained by last clusters A
    # first year first cluster is constrained by entire arbitrary UB
    @constraint(mp, [y in 2:n_years], W[y, 1] <= (A[y-1, length(K)] + elec.α_max * z_rep[y]) * T_k[y][1])
    @constraint(mp, [y in Y, k in 2:length(K)], W[y, k] <= A[y, k-1] * T_k[y][k])


    # demand constraint
    σ = 750 # kg/day
    D_y = 365 # days in a year
    @constraint(mp, [y in Y], sum(H[y, k] for k in K) >= σ * D_y)

    # electricity consumption
    @constraint(mp, [y in Y, k in K], E[y, k] == ϕ * γ * X_ON[y, k] + ϕ * γ * elec.ρ_sb * X_SB[y, k])

    
    @expression(mp, df[y in Y], (1/(1+ρ))^(y-1))
    @expression(mp, h_revenue[y in Y], sum(H[y, k] * elec.λ_H for k in K))
    @expression(mp, e_exp[y in Y], sum(E[y, k] * lmp_k[y][k] for k in K))
    # @expression(mp, npv_contrib, df[y]*(h_revenue - e_exp - λ_OPEX))
    @expression(mp, sp_obj[y in Y], df[y] * (e_exp[y] + elec.λ_OPEX - h_revenue[y]))

    # since this aggregate is a relaxation of the subproblem, this should be a valid LB on θ
    @constraint(mp, [y in Y], θ[y] >= sp_obj[y])

    # cutting planes
    # Benders optimality cuts
    M_cut = 1e9
    for y in Y
        A_start_expr = y == 1 ? elec.α_max : A[y-1,length(K)]
        for cut in cuts[y]
            rhs = cut.Q_LP + cut.π_dual * (A_start_expr - cut.A_prev)
            if cut.z_cut == 0
                @constraint(mp, θ[y] >= rhs - M_cut * z_rep[y])
            else
                @constraint(mp, θ[y] >= rhs - M_cut * (1 - z_rep[y]))
            end
        end
    end

    # integer feasibility cuts
    # ∑_{j∈S1}(1-z[j]) + ∑_{j∈S0}z[j] >= 1
    for cut in feas_cuts
        S1 = findall(==(1), cut.z_k)
        S0 = findall(==(0), cut.z_k)
        @constraint(mp,
            sum(1 - z_rep[j] for j in S1; init=0) +
            sum(z_rep[j]     for j in S0; init=0) >= 1)
    end

    # simple integer l-shaped cuts
    # active at z_k (θ >= Q_MIP), weakened to L_y for the opposite scenario
    for y in Y
        for cut in sl_cuts[y]
            if cut.z_k == 1 
                # if cut was derived from a replacement year (z_y=1 from master prob),
                # then this cut is saying that θ>= L_y if no longer a replacement year (defaults on loose cut)
                @constraint(mp, θ[y] >= cut.Q_MIP + (cut.L_y - cut.Q_MIP) * (1 - z_rep[y]))
                # otherwise θ >= Q_MIP and replacement was good
            # else
                # this cut SUCKS!!
                # @constraint(mp, θ[y] >= cut.Q_MIP + (cut.L_y - cut.Q_MIP) * z_rep[y])

            end
        end
    end

    # l-shaped integer cuts
    # θ >= Q_MIP + (L_y - Q_MIP) * (Σ_{j∈S1} (1-z_j) + Σ_{j∈S0} (z_j))
    for y in Y
        for cut in l_cuts[y]
            S1 = findall(==(1), cut.z_k)
            S0 = findall(==(0), cut.z_k)
            @constraint(mp, θ[y] >= cut.Q_MIP + (cut.L_y - cut.Q_MIP) * (sum(1 - z_rep[j] for j in S1; init=0) + sum(z_rep[j] for j in S0; init=0)))
        end
    end

    @expression(mp, replace_cost[y in Y], df_vec[y] * elec.λ_CAPEX_replace * z_rep[y])
    @objective(mp, Min, sum(θ[y] for y in Y) + sum(replace_cost[y] for y in Y))

    return mp
end



function build_subproblem(elec::Electrolyzer, y::Int, z_fixed_y::Int, A_init::Float64, lmp::Vector{Float64}; 
        w_α_max::Float64=elec.α_max, w_α_min::Float64=elec.α_min, ΔT::Float64 = 1.0, x_on_prev = 1.0, x_sb_prev = 0.0)


    sp_y = Model()

    # can produce tightening on w_t = a_t x^on_t
    T_length = length(lmp)   # 8760 for DAM, 35040 for RTM
    T   = 1:T_length
    df  = (1/(1+ρ))^(y-1)   # NPV discount factor for year y
    B = 9.66

    # Variables common to all years
    @variables sp_y begin
        e_tot[T]   >= 0
        h[T]       >= 0
        x_on[T],   Bin
        x_sb[T],   Bin
        x_off[T],  Bin
        x_start[T], Bin
        elec.α_min <= A[T] <= elec.α_max   # McCormick envelope is only valid on [α_min, α_max]
        elec.α_min <= A_prev <= elec.α_max
        w[T]       >= 0      # bilinear: A[t] * x_on[t]
    end

    # ── Energy consumption ────────────────────────────────────────────────────
    @constraint(sp_y, [t in T], e_tot[t] == elec.ϕ*x_on[t]*ΔT + elec.ϕ*x_sb[t]*ΔT*elec.ρ_sb)

    # ── Efficiency initial condition ──────────────────────────────────────────
    @constraint(sp_y, A_prev_fix, A_prev == A_init)
    @constraint(sp_y, A[1] == A_prev - A_prev * z_fixed_y + elec.α_max * z_fixed_y
                             - elec.δ_on*x_on[1]*ΔT - elec.δ_start*x_start[1])

    # ── Intra-year efficiency degradation ─────────────────────────────────────
    @constraint(sp_y, [t in 2:T_length], A[t] == A[t-1] - elec.δ_on*x_on[t]*ΔT - elec.δ_start*x_start[t])

    # ── Hydrogen production ───────────────────────────────────────────────────
    @constraint(sp_y, [t in T], h[t] == elec.ϕ*w[t]*ΔT + B*x_on[t]*ΔT)

    # McCormick envelope: w = A[t] * x_on[t]
    @constraint(sp_y, [t in T], w[t] <= w_α_max * x_on[t])
    @constraint(sp_y, [t in T], w[t] >= w_α_min * x_on[t])
    @constraint(sp_y, [t in T], w[t] >= A[t] - w_α_max*(1 - x_on[t]))
    @constraint(sp_y, [t in T], w[t] <= A[t] - w_α_min*(1 - x_on[t]))

    # ── 3-state logic ─────────────────────────────────────────────────────────
    @constraint(sp_y, [t in T], x_on[t] + x_off[t] + x_sb[t] == 1)

    # ── Daily hydrogen demand quota ───────────────────────────────────────────
    @constraint(sp_y, [d in 1:Int.(T_length//(24*n_per_hour))],
        sum(h[t] for t in ((d-1)*24*n_per_hour+1):(d*24*n_per_hour)) >= 750)

    # ── Startup logic (accounts for standby → on transitions) ─────────────────
    # x_on_prev_end/x_sb_prev_end carry terminal state from the prior year's MIP solve.
    # Defaults (0,0) are used for cut generation (conservative: always charges startup).
    @constraint(sp_y, x_start[1] >= x_on[1] - x_on_prev - x_sb_prev)
    @constraint(sp_y, [t in 2:T_length], x_start[t] >= x_on[t] - x_on[t-1] - x_sb[t-1])
    @constraint(sp_y, [t in 2:T_length], x_start[t] <= x_on[t])
    @constraint(sp_y, [t in 2:T_length], x_start[t] <= 1 - x_on[t-1] - x_sb[t-1])

    # ── Objective: discounted dispatch-only NPV for year y ────────────────────
    @expression(sp_y, h_revenue, sum(h[t]*elec.λ_H        for t in T))
    @expression(sp_y, e_exp,     sum(e_tot[t]*lmp[t] for t in T))
    @expression(sp_y, npv_contrib, df*(h_revenue - e_exp - λ_OPEX))

    @objective(sp_y, Min, -npv_contrib)

    return sp_y
end

function get_lp(m::JuMP.Model)
    lp_m = copy(m)
    relax_integrality(lp_m)
    @suppress set_optimizer(lp_m, Gurobi.Optimizer)
    set_optimizer_attribute(lp_m, "OutputFlag", false)
    @suppress optimize!(lp_m)
    
    status = termination_status(lp_m)
    if status != MOI.OPTIMAL && status != MOI.LOCALLY_SOLVED
        printf("m was not solved to optimality")
        return nothing
    end
    # printf("m was solved to optimality")
    return lp_m
end

function get_cut(m::JuMP.Model, z_cut::Int)
    Q_LP   = objective_value(m)
    A_prev = value(m[:A_prev])
    π_k    = dual(m[:A_prev_fix])
    return OptimalityCut(Q_LP, π_k, A_prev, z_cut)
end


function _get_x_prev(m::JuMP.Model)
    x_on_prev = value.(m[:x_on])[end]
    x_sb_prev = value.(m[:x_sb])[end]

    return x_on_prev, x_sb_prev
end

function sp_cache_key(y::Int, z_fixed_y::Int, A_init::Float64, x_on_prev::Float64, x_sb_prev::Float64)
    x_on_i = round(Int, x_on_prev)
    x_sb_i = round(Int, x_sb_prev)
    if z_fixed_y == 1
        # A_init cancels in the replacement reset constraint — subproblem solution is A_init-independent
        return (y, 1, x_on_i, x_sb_i)
    else
        return (y, 0, round(A_init, digits=8), x_on_i, x_sb_i)
    end
end


function get_L_y(elec::Electrolyzer, y::Int, lmp::Vector{Float64}; ΔT::Float64=1.0)
    # Global LB on Q_MIP(y, z, A) for all z ∈ {0,1} and A ∈ [α_min, α_max].
    # Uses LP relaxation at most-optimistic inputs:
    #   • A_init = α_max  (Q non-increasing in A for z=0; A-independent for z=1)
    #   • x_on_prev = 1   (system arrived ON → no forced startup at t=1)
    lp_z0 = get_lp(build_subproblem(elec, y, 0, elec.α_max, lmp;
                    w_α_max=elec.α_max, w_α_min=elec.α_min,
                    ΔT=ΔT, x_on_prev=1.0, x_sb_prev=0.0))
    lp_z1 = get_lp(build_subproblem(elec, y, 1, elec.α_max, lmp;
                    w_α_max=elec.α_max, w_α_min=elec.α_min,
                    ΔT=ΔT, x_on_prev=1.0, x_sb_prev=0.0))
    return min(objective_value(lp_z0), objective_value(lp_z1))
end

function traditional_benders(elec::Electrolyzer, n_years::Int;
                type="DAM", T_length=8760, k_max=60, ϵ=1e-3)

    """
    THis is aggregation over each year
    """

    get_data(n_years, T_length, type)
    get_electrolyzer_params(elec, n_years)

    cuts = [Vector{OptimalityCut}() for _ in 1:n_years]
    feas_cuts = FeasibilityCut[]
    sl_cuts = [Vector{SimpleIntegerLCut}() for _ in 1:n_years]
    l_cuts = [Vector{IntegerLCut}() for _ in 1:n_years]
    yearly_subprobs = Vector{Any}(undef, n_years)
    sp_cache = Dict{Any, SubproblemCacheEntry}()
    LB, UB = -Inf, Inf
    LBs = []
    UBs = []
    iter_times = Float64[]
    z_rep_final = zeros(Int, n_years)   # replacement schedule from last master solve
    avg_lmp = zeros(Float64, n_years)
    L_y_vec = zeros(Float64, n_years)

    # warm start the cuts
    x_on_prev = 0.0
    x_sb_prev = 0.0
    for y in 1:n_years
        z_fixed_y = 0
        sp_y_init = build_subproblem(elec, y, 0, elec.α_max, lmp[y],
                    x_on_prev = x_on_prev, x_sb_prev = x_sb_prev, ΔT = ΔT)

        lp_sp_y = get_lp(sp_y_init)

        cut_0a = get_cut(lp_sp_y, 0)
        push!(cuts[y], cut_0a)

        x_on_prev, x_sb_prev = _get_x_prev(lp_sp_y)

        L_y_vec[y] = get_L_y(elec, y, lmp[y], ΔT=ΔT)
        avg_lmp[y] = minimum(lmp[y])
    end

    # begin benders loop
    k=1
    while k <= k_max
        t_iter_start = time()
        # solve master problem
        master = build_masterproblem(elec, n_years, cuts, feas_cuts, sl_cuts, l_cuts)
        # @suppress set_optimizer_attribute(master, "MIPGap", 1e-4) # we are okay with a 0 mip gap here
        t_master      = @elapsed @suppress optimize!(master)
        z_fixed = round.(Int, value.(master[:z_rep]))
        @printf("Iter %-3d  [MP solved in %5.2fs]  z_rep = [%s]\n", k, t_master, join(string.(z_fixed), ", "))

        # update LB
        LB = max(LB, objective_value(master))
        push!(LBs, LB)

        # solve MIP subproblem
        A_init = elec.α_max
        x_on_prev = 0.0
        x_sb_prev = 0.0
        A_max = elec.α_max
        A_min = elec.α_min
        sp_obj = 0.0
        for y in 1:n_years
            z_fixed_y = z_fixed[y]
            w_max     = z_fixed_y == 1 ? elec.α_max : A_init
            cache_key = sp_cache_key(y, z_fixed_y, A_init, x_on_prev, x_sb_prev)

            if haskey(sp_cache, cache_key)
                entry   = sp_cache[cache_key]
                Q_MIP   = entry.Q_MIP
                sp_obj += Q_MIP
                x_on_prev, x_sb_prev = entry.x_on_end, entry.x_sb_end
                A_init  = entry.A_terminal
                @printf("\tSP y=%-2d  A_end=%6.4f  Q_MIP=%12.1f  [cached]\n", y, A_init, Q_MIP)
                # generate cuts against the current z_fixed even when MIP is cached
                push!(sl_cuts[y], SimpleIntegerLCut(Q_MIP, L_y_vec[y], z_fixed_y))
                push!(l_cuts[y],  IntegerLCut(Q_MIP, L_y_vec[y], z_fixed))
                continue
            end

            # build and solve MIP subproblem
            sp_y = build_subproblem(elec, y, z_fixed_y, A_init, lmp[y],
                                    w_α_max=w_max, w_α_min=A_min,
                                    x_on_prev=x_on_prev, x_sb_prev=x_sb_prev, ΔT = ΔT)
            @suppress set_optimizer(sp_y, Gurobi.Optimizer)
            @suppress set_optimizer_attribute(sp_y, "OutputFlag", false)
            @suppress set_optimizer_attribute(sp_y, "MIPGap", 1e-3)
            @suppress set_optimizer_attribute(sp_y, "TimeLimit", 300.0)
            t_sp = @elapsed @suppress optimize!(sp_y)
            sp_status = termination_status(sp_y)

            if sp_status ∈ (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
                Q_MIP = objective_value(sp_y)
                sp_obj += Q_MIP
                yearly_subprobs[y] = sp_y
                gap = MOI.get(sp_y, MOI.RelativeGap())
                @printf("\tSP y=%-2d  A_end=%6.4f  Q_MIP=%12.1f  t=%5.2fs gap=%3.3f\n", y, get_val(sp_y, :A)[end], Q_MIP, t_sp, gap)
            elseif sp_status == MOI.INFEASIBLE_OR_UNBOUNDED
                push!(feas_cuts, FeasibilityCut(collect(z_fixed)))
                @printf("  SP y=%d INFEASIBLE — no-good feasibility cut added (total=%d)\n",
                        y, length(feas_cuts))
                break
            elseif sp_status == MOI.INTERRUPTED
                println("Solver was interrupted.")
                if JuMP.result_count(sp_y) > 0
                    Q_MIP = objective_value(sp_y)
                    sp_obj += Q_MIP
                    yearly_subprobs[y] = sp_y
                    gap = MOI.get(sp_y, MOI.RelativeGap())
                    @printf("\tSP y=%-2d  A_end=%6.4f  Q_MIP=%12.1f  t=%5.2fs gap=%3.3f\n", y, get_val(sp_y, :A)[end], Q_MIP, t_sp, gap)
                end
            else
                @printf("  SP y=%d failed: %s\n", y, sp_status)
                break
            end

            # add l-shaped cuts
            L_y = L_y_vec[y]
            push!(sl_cuts[y], SimpleIntegerLCut(Q_MIP, L_y, z_fixed_y))
            push!(l_cuts[y],  IntegerLCut(Q_MIP, L_y, z_fixed))

            x_on_prev, x_sb_prev = _get_x_prev(sp_y)
            A_init = get_val(sp_y, :A)[end]

            sp_cache[cache_key] = SubproblemCacheEntry(Q_MIP, A_init, x_on_prev, x_sb_prev)
        end

        # iterate
        df = [(1 / (1 + elec.i))^(y - 1) for y in 1:n_years]
        UB_k = sp_obj + sum(df[y] * elec.λ_CAPEX_replace * z_fixed[y] for y in 1:n_years)
        UB = min(UB, UB_k)
        push!(UBs, UB)

        n_cuts = length(feas_cuts) + sum(length(cuts[y]) + length(sl_cuts[y]) + length(l_cuts[y]) for y in 1:n_years)
        gap    = abs(UB - LB) / max(abs(LB), 1.0)

        @printf("  LB=%14.1f  UB=%14.1f  Gap=%8.4f%%  Cuts=%d\n\n", LB, UB, gap * 100, n_cuts)

        push!(iter_times, time() - t_iter_start)
        if gap < ϵ
            @printf("Converged at iteration %d. Gap = %.4f%%\n", k, gap * 100)
            z_rep_final = z_fixed
            break
        end

        if k > 2
            if abs(UBs[end] - UBs[end-1]) < ϵ && abs(LBs[end] - LBs[end-1]) < ϵ
                @printf("No longer closing gap at iteration %d. Gap = %.4f%%\n", k, gap * 100)
                z_rep_final = z_fixed
                break
            end
        end
        k=k+1
    end
    return yearly_subprobs, cuts, feas_cuts, LBs, UBs, k, iter_times, z_rep_final

end


function aggregate_benders(elec::Electrolyzer, n_years::Int;
                type="DAM", T_length=8760, k_max=60, ϵ=1e-3)

    """
    THis is aggregation over each year
    """

    get_data(n_years, T_length, type)
    get_electrolyzer_params(elec, n_years)

    cuts = [Vector{OptimalityCut}() for _ in 1:n_years]
    feas_cuts = FeasibilityCut[]
    sl_cuts = [Vector{SimpleIntegerLCut}() for _ in 1:n_years]
    l_cuts = [Vector{IntegerLCut}() for _ in 1:n_years]
    yearly_subprobs = Vector{Any}(undef, n_years)
    sp_cache = Dict{Any, SubproblemCacheEntry}()
    LB, UB = -Inf, Inf
    LBs = []
    UBs = []
    iter_times = Float64[]
    z_rep_final = zeros(Int, n_years)   # replacement schedule from last master solve
    avg_lmp = zeros(Float64, n_years)
    L_y_vec = zeros(Float64, n_years)

    # warm start the cuts
    x_on_prev = 0.0
    x_sb_prev = 0.0
    for y in 1:n_years
        z_fixed_y = 0
        sp_y_init = build_subproblem(elec, y, 0, elec.α_max, lmp[y],
                    x_on_prev = x_on_prev, x_sb_prev = x_sb_prev, ΔT = ΔT)

        lp_sp_y = get_lp(sp_y_init)

        cut_0a = get_cut(lp_sp_y, 0)
        push!(cuts[y], cut_0a)

        x_on_prev, x_sb_prev = _get_x_prev(lp_sp_y)

        L_y_vec[y] = get_L_y(elec, y, lmp[y], ΔT=ΔT)
        avg_lmp[y] = minimum(lmp[y])
    end

    # begin benders loop
    k=1
    while k <= k_max
        t_iter_start = time()
        # solve master problem
        master = build_aggregate_masterproblem(elec, n_years, avg_lmp, cuts, feas_cuts, sl_cuts, l_cuts, type=type)
        @suppress set_optimizer_attribute(master, "MIPGap", 1e-4)
        t_master      = @elapsed @suppress optimize!(master)
        z_fixed = round.(Int, value.(master[:z_rep]))
        @printf("Iter %-3d  [MP solved in %5.2fs]  z_rep = [%s]\n", k, t_master, join(string.(z_fixed), ", "))

        # update LB
        LB = max(LB, objective_value(master))
        push!(LBs, LB)

        # solve MIP subproblem
        A_init = elec.α_max
        x_on_prev = 0.0
        x_sb_prev = 0.0
        A_max = elec.α_max
        A_min = elec.α_min
        sp_obj = 0.0
        for y in 1:n_years
            z_fixed_y = z_fixed[y]
            w_max     = z_fixed_y == 1 ? elec.α_max : A_init
            cache_key = sp_cache_key(y, z_fixed_y, A_init, x_on_prev, x_sb_prev)

            if haskey(sp_cache, cache_key)
                entry   = sp_cache[cache_key]
                Q_MIP   = entry.Q_MIP
                sp_obj += Q_MIP
                x_on_prev, x_sb_prev = entry.x_on_end, entry.x_sb_end
                A_init  = entry.A_terminal
                @printf("\tSP y=%-2d  A_end=%6.4f  Q_MIP=%12.1f  [cached]\n", y, A_init, Q_MIP)
                # generate cuts against the current z_fixed even when MIP is cached
                push!(sl_cuts[y], SimpleIntegerLCut(Q_MIP, L_y_vec[y], z_fixed_y))
                push!(l_cuts[y],  IntegerLCut(Q_MIP, L_y_vec[y], z_fixed))
                continue
            end

            # build and solve MIP subproblem
            sp_y = build_subproblem(elec, y, z_fixed_y, A_init, lmp[y],
                                    w_α_max=w_max, w_α_min=A_min,
                                    x_on_prev=x_on_prev, x_sb_prev=x_sb_prev, ΔT = ΔT)
            @suppress set_optimizer(sp_y, Gurobi.Optimizer)
            @suppress set_optimizer_attribute(sp_y, "OutputFlag", false)
            @suppress set_optimizer_attribute(sp_y, "MIPGap", 1e-3)
            @suppress set_optimizer_attribute(sp_y, "TimeLimit", 300.0)
            t_sp = @elapsed @suppress optimize!(sp_y)
            sp_status = termination_status(sp_y)

            if sp_status ∈ (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
                Q_MIP = objective_value(sp_y)
                sp_obj += Q_MIP
                yearly_subprobs[y] = sp_y
                gap = MOI.get(sp_y, MOI.RelativeGap())
                @printf("\tSP y=%-2d  A_end=%6.4f  Q_MIP=%12.1f  t=%5.2fs gap=%3.3f\n", y, get_val(sp_y, :A)[end], Q_MIP, t_sp, gap)
            elseif sp_status == MOI.INFEASIBLE_OR_UNBOUNDED
                push!(feas_cuts, FeasibilityCut(collect(z_fixed)))
                @printf("  SP y=%d INFEASIBLE — no-good feasibility cut added (total=%d)\n",
                        y, length(feas_cuts))
                break
            elseif sp_status == MOI.INTERRUPTED
                println("Solver was interrupted.")
                if JuMP.result_count(sp_y) > 0
                    Q_MIP = objective_value(sp_y)
                    sp_obj += Q_MIP
                    yearly_subprobs[y] = sp_y
                    gap = MOI.get(sp_y, MOI.RelativeGap())
                    @printf("\tSP y=%-2d  A_end=%6.4f  Q_MIP=%12.1f  t=%5.2fs gap=%3.3f\n", y, get_val(sp_y, :A)[end], Q_MIP, t_sp, gap)
                end
            else
                @printf("  SP y=%d failed: %s\n", y, sp_status)
                break
            end

            # add optimality cuts
            cut1 = get_cut(get_lp(build_subproblem(elec, y, z_fixed_y, A_init, lmp[y],
                                    w_α_max=w_max, w_α_min=A_min,
                                    x_on_prev=x_on_prev, x_sb_prev=x_sb_prev, ΔT = ΔT)), z_fixed_y)
            push!(cuts[y], cut1)
            # add l-shaped cuts
            L_y = L_y_vec[y]
            push!(sl_cuts[y], SimpleIntegerLCut(Q_MIP, L_y, z_fixed_y))
            push!(l_cuts[y],  IntegerLCut(Q_MIP, L_y, z_fixed))

            x_on_prev, x_sb_prev = _get_x_prev(sp_y)
            A_init = get_val(sp_y, :A)[end]

            sp_cache[cache_key] = SubproblemCacheEntry(Q_MIP, A_init, x_on_prev, x_sb_prev)
        end

        # iterate
        df = [(1 / (1 + elec.i))^(y - 1) for y in 1:n_years]
        UB_k = sp_obj + sum(df[y] * elec.λ_CAPEX_replace * z_fixed[y] for y in 1:n_years)
        UB = min(UB, UB_k)
        push!(UBs, UB)

        n_cuts = length(feas_cuts) + sum(length(cuts[y]) + length(sl_cuts[y]) + length(l_cuts[y]) for y in 1:n_years)
        gap    = abs(UB - LB) / max(abs(LB), 1.0)

        @printf("  LB=%14.1f  UB=%14.1f  Gap=%8.4f%%  Cuts=%d\n\n", LB, UB, gap * 100, n_cuts)

        push!(iter_times, time() - t_iter_start)
        if gap < ϵ
            @printf("Converged at iteration %d. Gap = %.4f%%\n", k, gap * 100)
            z_rep_final = z_fixed
            break
        end

        if k > 2
            if abs(UBs[end] - UBs[end-1]) < ϵ && abs(LBs[end] - LBs[end-1]) < ϵ
                @printf("No longer closing gap at iteration %d. Gap = %.4f%%\n", k, gap * 100)
                z_rep_final = z_fixed
                break
            end
        end
        k=k+1
    end
    return yearly_subprobs, cuts, feas_cuts, LBs, UBs, k, iter_times, z_rep_final
end


function aggregate_k_benders(elec::Electrolyzer, n_years::Int, k_clusters::Int;
                type="DAM", T_length=8760, k_max=60, ϵ=1e-3)
    """
    This is aggregation over k clusters per year
    There were some ideas to generalize k clusters over the entire time horizon 
    but then I am not sure how replacement would work. I think hierarchical structure is valuable here
    (especially given the time constraints of getting this paper out)
    """

    get_data(n_years, T_length, type)
    get_electrolyzer_params(elec, n_years)

    cuts = [Vector{OptimalityCut}() for _ in 1:n_years]
    feas_cuts = FeasibilityCut[]
    sl_cuts = [Vector{SimpleIntegerLCut}() for _ in 1:n_years]
    l_cuts = [Vector{IntegerLCut}() for _ in 1:n_years]
    yearly_subprobs = Vector{Any}(undef, n_years)
    sp_cache = Dict{Any, SubproblemCacheEntry}()
    LB, UB = -Inf, Inf
    LBs = []
    UBs = []
    iter_times = Float64[]
    z_rep_final = zeros(Int, n_years)   # replacement schedule from last master solve
    avg_lmp = zeros(Float64, n_years)
    L_y_vec = zeros(Float64, n_years)

    # warm start the cuts
    x_on_prev = 0.0
    x_sb_prev = 0.0
    for y in 1:n_years
        z_fixed_y = 0
        sp_y_init = build_subproblem(elec, y, 0, elec.α_max, lmp[y],
                    x_on_prev = x_on_prev, x_sb_prev = x_sb_prev, ΔT = ΔT)

        lp_sp_y = get_lp(sp_y_init)

        cut_0a = get_cut(lp_sp_y, 0)
        push!(cuts[y], cut_0a)

        x_on_prev, x_sb_prev = _get_x_prev(lp_sp_y)

        L_y_vec[y] = get_L_y(elec, y, lmp[y], ΔT=ΔT)
        avg_lmp[y] = minimum(lmp[y])
    end

    # begin benders loop
    k=1
    while k <= k_max
        t_iter_start = time()
        # solve master problem
        master = build_k_aggregate_masterproblem(elec, n_years, k_clusters, lmp, cuts, feas_cuts, sl_cuts, l_cuts, type=type)
        @suppress set_optimizer_attribute(master, "MIPGap", 1e-4)
        t_master      = @elapsed @suppress optimize!(master)
        z_fixed = round.(Int, value.(master[:z_rep]))
        @printf("Iter %-3d  [MP solved in %5.2fs]  z_rep = [%s]\n", k, t_master, join(string.(z_fixed), ", "))

        # update LB
        LB = max(LB, objective_value(master))
        push!(LBs, LB)

        # solve MIP subproblem
        A_init = elec.α_max
        x_on_prev = 0.0
        x_sb_prev = 0.0
        A_max = elec.α_max
        A_min = elec.α_min
        sp_obj = 0.0
        ub_valid = true

        for y in 1:n_years
            z_fixed_y = z_fixed[y]
            w_max     = z_fixed_y == 1 ? elec.α_max : A_init
            cache_key = sp_cache_key(y, z_fixed_y, A_init, x_on_prev, x_sb_prev)

            if haskey(sp_cache, cache_key)
                entry   = sp_cache[cache_key]
                Q_MIP   = entry.Q_MIP
                sp_obj += Q_MIP
                x_on_prev, x_sb_prev = entry.x_on_end, entry.x_sb_end
                A_init  = entry.A_terminal
                @printf("\tSP y=%-2d  A_end=%6.4f  Q_MIP=%12.1f  [cached]\n", y, A_init, Q_MIP)
                # generate cuts against the current z_fixed even when MIP is cached
                push!(sl_cuts[y], SimpleIntegerLCut(Q_MIP, L_y_vec[y], z_fixed_y))
                push!(l_cuts[y],  IntegerLCut(Q_MIP, L_y_vec[y], z_fixed))
                continue
            end

            # build and solve MIP subproblem
            sp_y = build_subproblem(elec, y, z_fixed_y, A_init, lmp[y],
                                    w_α_max=w_max, w_α_min=A_min,
                                    x_on_prev=x_on_prev, x_sb_prev=x_sb_prev, ΔT = ΔT)
            @suppress set_optimizer(sp_y, Gurobi.Optimizer)
            # if y != 8
                # @suppress set_optimizer_attribute(sp_y, "OutputFlag", false)
            # end
            @suppress set_optimizer_attribute(sp_y, "OutputFlag", false)
            @suppress set_optimizer_attribute(sp_y, "MIPGap", 1e-3)
            @suppress set_optimizer_attribute(sp_y, "TimeLimit", 300.0)
            t_sp = @elapsed @suppress optimize!(sp_y)
            sp_status = termination_status(sp_y)
            
            if sp_status ∈ (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
                Q_MIP = objective_value(sp_y)
                sp_obj += Q_MIP
                yearly_subprobs[y] = sp_y
                gap = MOI.get(sp_y, MOI.RelativeGap())
                @printf("\tSP y=%-2d  A_end=%6.4f  Q_MIP=%12.1f  t=%5.2fs gap=%3.3f\n", y, get_val(sp_y, :A)[end], Q_MIP, t_sp, gap)
            elseif sp_status == MOI.INFEASIBLE_OR_UNBOUNDED
                push!(feas_cuts, FeasibilityCut(collect(z_fixed)))
                @printf("  SP y=%d INFEASIBLE — no-good feasibility cut added (total=%d)\n",
                        y, length(feas_cuts))
                break
            elseif sp_status ∈ (MOI.INTERRUPTED, MOI.TIME_LIMIT)
                println("Solver was interrupted or hit the time limit.")
                if JuMP.result_count(sp_y) > 0
                    # Retrieve your variables or objective value
                    Q_MIP = objective_value(sp_y)
                    sp_obj += Q_MIP
                    yearly_subprobs[y] = sp_y
                    gap = MOI.get(sp_y, MOI.RelativeGap())
                    @printf("\tSP y=%-2d  A_end=%6.4f  Q_MIP=%12.1f  t=%5.2fs gap=%3.3f\n", y, get_val(sp_y, :A)[end], Q_MIP, t_sp, gap)
                else
                    print("\t No solution found.\n")
                    break
                end
            else
                @printf("  SP y=%d failed: %s\n", y, sp_status)
                ub_valid = false
                break
            end

            # add optimality cuts
            cut1 = get_cut(get_lp(build_subproblem(elec, y, z_fixed_y, A_init, lmp[y],
                                    w_α_max=w_max, w_α_min=A_min,
                                    x_on_prev=x_on_prev, x_sb_prev=x_sb_prev, ΔT = ΔT)), z_fixed_y)
            push!(cuts[y], cut1)

            L_y = L_y_vec[y]
            push!(sl_cuts[y], SimpleIntegerLCut(Q_MIP, L_y, z_fixed_y))
            push!(l_cuts[y],  IntegerLCut(Q_MIP, L_y, z_fixed))

            x_on_prev, x_sb_prev = _get_x_prev(sp_y)
            A_init = get_val(sp_y, :A)[end]

            sp_cache[cache_key] = SubproblemCacheEntry(Q_MIP, A_init, x_on_prev, x_sb_prev)
        end

        # iterate
        df = [(1 / (1 + elec.i))^(y - 1) for y in 1:n_years]
        if ub_valid
            UB_k = sp_obj + sum(df[y] * elec.λ_CAPEX_replace * z_fixed[y] for y in 1:n_years)
            UB = min(UB, UB_k)
        end
        push!(UBs, UB)

        n_cuts = length(feas_cuts) + sum(length(cuts[y]) + length(sl_cuts[y]) + length(l_cuts[y]) for y in 1:n_years)
        gap    = abs(UB - LB) / max(abs(LB), 1.0)

        @printf("  LB=%14.1f  UB=%14.1f  Gap=%8.4f%%  Cuts=%d\n\n", LB, UB, gap * 100, n_cuts)

        push!(iter_times, time() - t_iter_start)
        if gap < ϵ
            @printf("Converged at iteration %d. Gap = %.4f%%\n", k, gap * 100)
            z_rep_final = z_fixed
            break
        end

        if k > 2
            if abs(UBs[end] - UBs[end-1]) < ϵ && abs(LBs[end] - LBs[end-1]) < ϵ
                @printf("No longer closing gap at iteration %d. Gap = %.4f%%\n", k, gap * 100)
                z_rep_final = z_fixed
                break
            end
        end
        k=k+1
    end
    return yearly_subprobs, cuts, feas_cuts, LBs, UBs, k, iter_times, z_rep_final
end