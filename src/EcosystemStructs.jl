
using UUIDs, JSON

Base.@kwdef mutable struct HiddenAgent{T} <: Agent
    id::Int64
    name::String
    model::T
    in_shape::Tuple
    out_shape::Tuple
    data::Array{Float32, N} where N
    output::AbstractArray = model(data)
end

Base.@kwdef mutable struct InputAgent <: Agent
    id::Int64
    name::String
    shape::Tuple
    data::Array{Float32, N} where N
    shapley::AbstractFloat = 0.0
end

Base.@kwdef mutable struct Network{T} <: Component
    id::Int64
    name::String
    model::T
    in_shape::Tuple
    out_shape::Tuple
    data::Array{Float32, N} where N
    output::AbstractArray = model(data)
end

function set_component_data!(c::Component, data::Array{Float32, N}) where {N}
    c.data = data
    c.output = c.model(data)
end

thename(::Type{T}) where {T} = eval(nameof(T))

function Base.show(io::IO, a::Component)
    println(io, 
            json(Dict(
            :id => string(a.id),
            :name => a.name,
            :type => thename(typeof(a)),
            :shapley => hasproperty(a, :shapley) ? a.shapley : Nothing
        ), 4)
    )
end
