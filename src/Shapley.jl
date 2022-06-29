
using Combinatorics

@doc """
The Shapley algorithm (solution concept in cooperative game theory) for computing 
the Shapley values of a set of agents. The Shapley values are the expected 
utility of each agent in a set.
""" ->
function shapley(eco::Ecosystem, model::T, 
        data::Any, labels::Any, τ::Float64) where T <: Function 
    N = collect(powerset(eco.ii))
    Φ = Dict{Int64, Float64}()
    ν = []
    for ii in keys(eco.ii)
        ϕ = 0
        tmp_ν = Vector{Float64}()
        for S in N

            idxs = map(x->eco.ii[x], S)
            d = deepcopy(data)
            [d[i, :] *= 0 for i=1:size(d)[1] if i ∉ idxs];
            m₍si₎ = mean((map(x->model(x) > τ, eachcol(d))) .== labels)

            deleteat!(S, findall(x->x==ii, S))
            idxs = map(x->eco.ii[x], S)
            d = deepcopy(data)
            [d[i, :] *= 0 for i=1:size(d)[1] if i ∉ idxs];
            m₍s₎ = mean((map(x->model(x), eachcol(d)) .> τ) .== labels)

            α = factorial(length(S))*factorial(length(eco.ii)-length(S)-1)/factorial(length(eco.ii))

            ϕ += (α * (m₍si₎ - m₍s₎))
            append!(tmp_ν, m₍si₎ - m₍s₎)

        end
        Φ[ii] = ϕ
        append!(ν, [tmp_ν])
    end
    return Φ, ν
end
