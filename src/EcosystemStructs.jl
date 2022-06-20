
using JSON

mutable struct Component{T <: ComponentType} <: AbstractComponent{T}
    id::Int64
    name::String
    model::Any
    shapley::AbstractFloat
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
