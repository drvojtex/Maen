
using Combinatorics, StatsBase
using Graphs, SimpleWeightedGraphs
using HypothesisTests

@doc """
The Shapley algorithm (solution concept in cooperative game theory) for computing 
the Shapley values of a set of agents. The Shapley values are the expected 
utility of each agent in a set.
""" ->
function shapley(eco::Ecosystem, model::T, 
        data::Any, labels::Any, τ::Float64) where T <: Function 
    N = collect(powerset(collect(keys(eco.ii))))
    Φ = Dict{Int64, Float64}()
    ν = Dict{Int64, Vector{Float64}}()
    for ii in keys(eco.ii)
        ϕ = 0
        tmp_ν = Vector{Float64}()
        for S in N

            m₍si₎ = get_effort(data, labels, S, τ, eco, model)
            deleteat!(S, findall(x->x==ii, S))
            m₍s₎ = get_effort(data, labels, S, τ, eco, model)

            α₁ = factorial(length(S))
            α₂ = factorial(length(eco.ii)-length(S)-1)
            α₃ = factorial(length(eco.ii))
            α = α₁ * α₂ / α₃

            ϕ += (α * (m₍si₎ - m₍s₎))
            append!(tmp_ν, m₍si₎ - m₍s₎)

        end
        Φ[ii] = ϕ
        ν[ii] = tmp_ν
    end
    return Φ, ν
end

function get_effort(data::Any, labels::Any, S::T, τ::Float64,
        eco::Ecosystem, model::F) where {T <: AbstractVector, F <: Function}
    idxs = map(x->eco.ii[x], S)
    d = deepcopy(data)
    [d[i, :] *= 0 for i=1:size(d)[1] if i ∉ idxs];
    mean((map(x->model(x) > τ, eachcol(d))) .== labels)
end

function inputs_relations_graph(efforts::Dict{Int64, Vector{Float64}})
    g = SimpleWeightedGraph{Int64, Float64}(length(efforts))
    for c in combinations(collect(keys(efforts)), 2)
        add_edge!(g, c[1], c[2], 1/pvalue(SignedRankTest(efforts[c[1]], efforts[c[2]])))
    end
    return g
end
