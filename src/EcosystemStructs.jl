
using JSON

Base.@kwdef mutable struct HiddenAgent{T} <: Agent
    id::Int64
    name::String
    model::T
end

let input_id::Int = 0
    mutable struct InputAgent <: Agent
        id::Int64
        input_id::Int64
        name::String
        model::Function
        shapley::AbstractFloat
        function InputAgent(id::Int64, name::String)
            input_id += 1
            new(id, input_id, name, identity, 0.0)
        end
    end
end

Base.@kwdef mutable struct Network{T} <: Component
    id::Int64
    name::String
    model::T
end

HiddenAgent(n::Network) = HiddenAgent(id=n.id, name=n.name, model=n.model)
Network(ha::HiddenAgent) = Network(id=ha.id, name=ha.name, model=ha.model)

thetype(::Type{T}) where {T} = eval(nameof(T))

function Base.show(io::IO, a::Component)
    println(io, 
            json(Dict(
            :id => string(a.id),
            :name => a.name,
            :type => thetype(typeof(a)),
            :shapley => hasproperty(a, :shapley) ? a.shapley : Nothing
        ), 4)
    )
end
