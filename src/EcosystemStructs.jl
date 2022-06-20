
using JSON
using Graphs

mutable struct Component{T <: ComponentType} <: AbstractComponent{T}
    id::Int64
    name::String
    model::Any
    shapley::AbstractFloat
end

struct Ecosystem
    g::SimpleDiGraph{Int64}
    schc::Vector{Component}
    ii::Vector{Int64}
    sch::Vector{Int64}
end

function Base.show(io::IO, a::Component)
    println(io, 
            json(Dict(
            :id => string(a.id),
            :name => a.name,
            :type => typeof(a).parameters[1],
            :shapley => hasproperty(a, :shapley) ? a.shapley : Nothing
        ), 4)
    )
end
