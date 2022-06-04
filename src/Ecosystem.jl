
using UUIDs

Base.@kwdef mutable struct Agent{T} <: Component
    id::UUID = UUIDs.uuid4()
    model::T
    name::String
    ps::Dict{Symbol, Vector} = Dict{Symbol, Vector}()
    data::Array{Float64, N} where N
    shapley::AbstractFloat = 0.0
    output::AbstractArray = model(data)
end

Base.@kwdef mutable struct Network{T}  <: Component
    model::T
    name::String
    ps::Dict{Symbol, Vector} = Dict{Symbol, Vector}()
    shapley::AbstractFloat = 0.0
end

thename(::Type{T}) where {T} = eval(nameof(T))

function Base.show(io::IO, a::Component)
    type::UnionAll = thename(typeof(a))
    name::String = a.name
    shapley::AbstractFloat = a.shapley
    print(io, "component: $type\nname: $name\nshapley: $shapley \n")
end
