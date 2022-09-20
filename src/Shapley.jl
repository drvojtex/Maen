
using Combinatorics, StatsBase
using Graphs, SimpleWeightedGraphs
using HypothesisTests
using ThreadTools


@doc """
The Shapley algorithm (solution concept in cooperative game theory) for computing 
the Shapley values of a set of agents. The Shapley values are the expected 
utility of each agent in a set.
""" ->
function shapley(eco::Ecosystem, data::Any, labels::Any, τ::Float64, 
    ids::Vector{Int64}, noise_method::Bool)

    all_ids::Vector{Int64} = map(x->x.id, values(eco.comps))
    N::Vector{Vector{Int64}} = collect(powerset(ids))[2:end]
    Φ = Dict{Int64, Float64}()
    ν = Dict{Int64, Vector{Float64}}()

    @show ids
    for cid in ids
        @show cid
        ϕ::Float64 = .0
        tmp_ν = Vector{Float64}()
        for S in N 
            
            S = deepcopy(S)
            λ = length(setdiff(S, cid))

            union!(S, setdiff(all_ids, ids))
            
            # TODO: argmax vs treshold

            m₍si₎::Float64 = mean(
                (tmap(x -> argmax(subset_model(
                    eco, x, S, noise=noise_method)[end])-1, data)) .== labels
            )
            setdiff!(S, cid)
            m₍s₎::Float64 = mean(
                (tmap(x -> argmax(subset_model(
                    eco, x, S, noise=noise_method)[end])-1, data)) .== labels
            )

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
    return Φ, relations_graph(ν)
end

function hiddenagents_shapley(eco::Ecosystem, data::Any, labels::Any; τ::Float64=.5)
    ids = map(x->x.id, filter(x->typeof(x).parameters[1]==HiddenAgent, collect(values(eco.comps))))
    return shapley(eco, data, labels, τ, ids, true)
end

function inputagents_shapley(eco::Ecosystem, data::Any, labels::Any; τ::Float64=.5)
    ids = map(x->x.id, filter(x->typeof(x).parameters[1]==InputAgent, collect(values(eco.comps))))
    return shapley(eco, data, labels, τ, ids, false)
end

function relations_graph(efforts::Dict{Int64, Vector{Float64}})
    g = SimpleWeightedGraph{Int64, Float64}(length(efforts))
    for c in combinations(collect(keys(efforts)), 2)
        add_edge!(g, c[1], c[2], 1/pvalue(SignedRankTest(efforts[c[1]], efforts[c[2]])))
    end
    return g
end

function cluster_relations(g::SimpleWeightedGraph{Int64, Float64}; α::Float64=.0)
    α = α == .0 ? mapreduce(e -> e.weight, +, kruskal_mst(g))/length(vertices(g)) : α
    irg = SimpleWeightedGraph(length(vertices(g)))
    map(e -> add_edge!(irg, e.src, e.dst, e.weight), filter(e -> e.weight < α,  kruskal_mst(g)))
    connected_components(irg)
end
