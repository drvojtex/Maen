
using JSON
using Graphs
using DocStringExtensions

"""
The struct of the neural network graph. 

$(FIELDS)
"""
Base.@kwdef mutable struct Network
    "Identifiers of the nodes."
    ids::Vector{Int64} = []
    "Data types of the nodes (InputAgent, HiddenAgent, OutputAgent)."
    types::Vector{String} = []
    "Names of the nodes."
    names::Vector{String} = []
    "Adjacency list of directed connections between nodes."
    adjacency_list::Dict{Int64, Vector{Int64}} = Dict()
end

"""
The struct of the neural network component. 
It can be of the data type InputAgent, HiddenAgent or OutputAgent.

$(FIELDS)
"""
mutable struct Component{T <: ComponentType} <: AbstractComponent{T}
    "Identifier of the component in the network schema."
    id::Int64
    "Human readeble name of the component."
    name::String
    "Function defining the math model of the component."
    model::Any
    "Score of the component's importance in the schema"
    importance::AbstractFloat
end

"""
The struct of the neural network ecosystem. 

$(FIELDS)
"""
mutable struct Ecosystem
    "Graph which defines the connection of components."
    g::SimpleDiGraph{Int64}
    "Scheduled components - order of propagation trought the model schema."
    schc::Vector{Component}
    "Mapping of components to sample particular value 
        (key, value) = (input component id, order in sample)."
    ii::Dict{Int64, Int64}
    "Scheduled ids of components - order of propagation trought the model schema."
    sch::Vector{Int64}
    "Dictionary of all components (key, value) = (id, component)."
    comps::Dict{String, Component}
    "Treinable parameters of the components."
    ps_obj::Any
end

function Base.show(io::IO, a::Component)
    println(io, 
            json(Dict(
            :id => string(a.id),
            :name => a.name,
            :type => typeof(a).parameters[1],
            :importance => hasproperty(a, :importance) ? a.importance : Nothing
        ), 4)
    )
end
