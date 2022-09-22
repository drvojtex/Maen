
using Combinatorics, StatsBase
using Graphs, SimpleWeightedGraphs
using HypothesisTests
using ThreadTools, ProgressBars, Printf


@doc """
The Shapley algorithm (solution concept in cooperative game theory) for computing 
the Shapley values of a set of agents. The Shapley values are the expected 
utility of each agent in a set.

julia> shapley(
    eco, data, labels, ids, (x, S) -> argmax(
    subset_model(eco, x, S, noise=true)[end]) - 1
)

""" ->
function shapley(eco::Ecosystem, data::Any, labels::Any, 
    ids::Vector{Int64}, s_model::Function)

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

            m₍si₎::Float64 = mean(
                (tmap(x -> s_model(x, S), data)) .== labels
            )
            setdiff!(S, cid)
            m₍s₎::Float64 = mean(
                (tmap(x -> s_model(x, S), data)) .== labels
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

@doc """
julia> hiddenagents_shapley(eco, data, labels, 
    (eco, x, S) -> argmax(subset_model(eco, x, S, noise=true)[end]) - 1
)
""" ->
function hiddenagents_shapley(eco::Ecosystem, data::Any, labels::Any, s_model::Function)
    ids = map(x->x.id, filter(x->typeof(x).parameters[1]==HiddenAgent, collect(values(eco.comps))))
    return shapley(eco, data, labels, ids, (x, S) -> s_model(eco, x, S))
end

@doc """
julia> hiddenagents_shapley(eco, data, labels, 
    (eco, x, S) -> argmax(subset_model(eco, x, S, noise=false)[end]) - 1
)
""" ->
function inputagents_shapley(eco::Ecosystem, data::Any, labels::Any)
    ids = map(x->x.id, filter(x->typeof(x).parameters[1]==InputAgent, collect(values(eco.comps))))
    return shapley(eco, data, labels, ids, (x, S) -> s_model(eco, x, S))
end

function relations_graph(efforts::Dict{Int64, Vector{Float64}})
    g = SimpleWeightedGraph{Int64, Float64}(length(efforts))
    for c in combinations(collect(keys(efforts)), 2)
        add_edge!(g, c[1], c[2], pvalue(SignedRankTest(efforts[c[1]], efforts[c[2]])))
    end
    return g
end
