using DataFrames, XLSX
include("electrolyzer_struct.jl")

function get_data(n_years::Int=20, T_length::Int=8760, type::String="DAM")
    if n_years > 40
        error("too many years")
    end
    # global n_years = n_years

    if type == "DAM"
        dam_11yr = DataFrame(XLSX.readtable((@__DIR__) * "/../data/ERCOT_DAM_AVG_2014-2024.xlsx", "Sheet1"))
        lmp_dam_10 = Float64.(dam_11yr[1:8760*10, "SettlementPointPrice"])
        lmp_dam_20 = vcat(lmp_dam_10, lmp_dam_10)

        if n_years > 20
            lmp_dam_20 = vcat(lmp_dam_20, lmp_dam_20)
        end

        global n_per_hour = 1
        global ΔT = 1.0
        global lmp = [lmp_dam_20[(y-1)*T_length+1:y*T_length] for y in 1:n_years]
        global full_lmp = lmp_dam_20[1:T_length*n_years]
        # avg_lmp_y = [sum(lmp_rtm[y]) / length(lmp_rtm[y]) for y in 1:n_years]
    elseif type == "RTM"
        rtm_11yr = DataFrame(XLSX.readtable((@__DIR__) * "/../data/ERCOT_15RTM_2014-2024.xlsx", "Sheet1"))
        lmp_rtm_11 = Float64.(rtm_11yr[1:8760*4*11, "Settlement Point Price"])
        lmp_rtm_22 = vcat(lmp_rtm_11, lmp_rtm_11)

        if n_years > 22
            lmp_rtm_22 = vcat(lmp_rtm_22, lmp_rtm_22)
        end


        global lmp = [lmp_rtm_22[(y-1)*T_length*4+1:y*T_length*4] for y in 1:n_years]
        global n_per_hour = 4
        global ΔT = 1.0 / n_per_hour
        global full_lmp = lmp_rtm_22[1:T_length*n_per_hour*n_years]
    end
end


function get_electrolyzer_params(θ::Electrolyzer, n_years)
    global ϕ = θ.ϕ
    global α_max = θ.α_max
    global α_init = θ.α_max
    global α_min = θ.α_min
    global δ_on = θ.δ_on
    global δ_start = θ.δ_start
    global ρ = θ.i
    global ρ_sb = θ.ρ_sb
    global B = 9.66
    global total_days = 365
    global D_H = 750.0          # kg/day demand
    global λ_H = θ.λ_H
    global λ_OPEX = θ.λ_OPEX
    global λ_CAPEX_replace = θ.λ_CAPEX_Stack
    global λ_CAPEX = θ.λ_CAPEX_Plant
    global df_vec = [(1 / (1 + ρ))^(y - 1) for y in 1:n_years]
    # global δ_annual = δ_on * 8760 # approximate degrad in one year of operation
    global stack_lifetime = Int(round(θ.ℓ)) # device lifetime in integer years
end

function get_val(model, var::Symbol)
    return vec(float.(Array(value.(model[var]))))
end