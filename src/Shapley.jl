
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
        ϕ::Float64 = .0
        tmp_ν = Vector{Float64}()
        for S in N

            m₍si₎::Float64 = get_effort(data, labels, S, τ, eco, model)
            deleteat!(S, findall(x->x==ii, S))
            m₍s₎::Float64 = get_effort(data, labels, S, τ, eco, model)

            γ₁::Int64 = factorial(length(S))
            γ₂::Int64 = factorial(length(eco.ii)-length(S)-1)
            γ₃::Int64 = factorial(length(eco.ii))
            γ::Float64 = γ₁ * γ₂ / γ₃

            ϕ += (γ * (m₍si₎ - m₍s₎))
            append!(tmp_ν, m₍si₎ - m₍s₎)

        end
        Φ[ii] = ϕ
        ν[ii] = tmp_ν
    end
    return Φ, inputs_relations_graph(ν)
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

function cluster_inputs_relations(g::SimpleWeightedGraph{Int64, Float64}; α::Float64=.0)
    α = α == .0 ? mapreduce(e -> e.weight, +, kruskal_mst(g))/length(vertices(g)) : α
    irg = SimpleWeightedGraph(length(vertices(g)))
    map(e -> add_edge!(irg, e.src, e.dst, e.weight), filter(e -> e.weight < α,  kruskal_mst(g)))
    connected_components(irg)
end
