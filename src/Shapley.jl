
using Combinatorics, StatsBase
using Graphs, SimpleWeightedGraphs
using HypothesisTests
using ThreadTools, ProgressBars, Printf


function shapley(all_ids::Vector{Int64}, ids::Vector{Int64}, subset_acc::Function)

    shap = Dict{Int64, Float64}()

    for cid in ids
        @printf "cid %d (idx: %d), length of ids: %d\n" cid findall(x->x==cid, ids)[1] length(ids)

        ϕ::Float64 = .0

        N::Vector{Vector{Int64}} = collect(powerset(
            setdiff(ids, cid)
        ))[2:end]
        
        for S in ProgressBar(N) 

            λ::Int64 = length(S)
            union!(S, setdiff(all_ids, ids))

            m₍si₎::Float64 = subset_acc(union(S, cid))
            m₍s₎::Float64 = subset_acc(S)

            γ₁::Int64 = factorial(λ)
            γ₂::Int64 = factorial(length(ids)-λ-1)
            γ₃::Int64 = factorial(length(ids))
            γ::Float64 = γ₁ * γ₂ / γ₃

            ϕ += (γ * (m₍si₎ - m₍s₎))
        end 
        shap[cid] = ϕ
    end
    return shap
end

function hiddenagents_shapley(eco::Ecosystem, data::Any, labels::Any, subset_acc::Function)
    ids::Vector{Int64} = map(x->x.id, filter(x->typeof(x).parameters[1]==HiddenAgent, collect(values(eco.comps))))
    shapley(map(x->x.id, values(eco.comps)), ids, (S) -> subset_acc(S, data, labels))
end

function inputagents_shapley(eco::Ecosystem, data::Any, labels::Any, subset_acc::Function)
    ids::Vector{Int64} = map(x->x.id, filter(x->typeof(x).parameters[1]==InputAgent, collect(values(eco.comps))))
    shapley(map(x->x.id, values(eco.comps)), ids, (S) -> subset_acc(S, data, labels))
end
