
using Combinatorics, StatsBase
using Graphs, SimpleWeightedGraphs
using HypothesisTests
using ThreadTools, ProgressBars, Printf


function shapley(eco::Ecosystem, data::Any, labels::Any, 
    ids::Vector{Int64}, subsec_acc::Function)

    all_ids::Vector{Int64} = map(x->x.id, values(eco.comps))
    N::Vector{Vector{Int64}} = collect(powerset(ids))[2:end]
    Φ = Dict{Int64, Float64}()
    ν = Dict{Int64, Vector{Float64}}()

    for cid in ids
        @printf "cid %d (idx: %d), length of ids: %d\n" cid findall(x->x==cid, ids)[1] length(ids)

        ϕ::Float64 = .0
        tmp_ν = Vector{Float64}()
        
        for S in ProgressBar(N) 
            
            S = deepcopy(S)
            λ = length(setdiff(S, cid))

            union!(S, setdiff(all_ids, ids))

            m₍si₎::Float64 = subsec_acc(S, data, labels)
            setdiff!(S, cid)
            m₍s₎::Float64 = subsec_acc(S, data, labels)

            γ₁::Int64 = factorial(λ)
            γ₂::Int64 = factorial(length(ids)-λ-1)
            γ₃::Int64 = factorial(length(ids))
            γ::Float64 = γ₁ * γ₂ / γ₃

            ϕ += (γ * (m₍si₎ - m₍s₎))
            append!(tmp_ν, m₍si₎ - m₍s₎)

        end 
        Φ[cid] = ϕ
        ν[cid] = tmp_ν
    end
    return Φ
end

function hiddenagents_shapley(eco::Ecosystem, data::Any, labels::Any, subsec_acc::Function)
    ids = map(x->x.id, filter(x->typeof(x).parameters[1]==HiddenAgent, collect(values(eco.comps))))
    return shapley(eco, data, labels, ids, (S, d, l) -> subsec_acc(S, d, l))
end

function inputagents_shapley(eco::Ecosystem, data::Any, labels::Any, subsec_acc::Function)
    ids = map(x->x.id, filter(x->typeof(x).parameters[1]==InputAgent, collect(values(eco.comps))))
    return shapley(eco, data, labels, ids, (S, d, l) -> subsec_acc(S, d, l))
end
