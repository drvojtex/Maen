
module Maen

export Component, Ecosystem
export InputAgent, HiddenAgent, NetworkAgent

export create_ecosystem, scv, scheduling
export model, agents_dims
export ecosystem_shapley, inputagents_cluster_relations
export simple_visu

abstract type AbstractComponent{T} end
abstract type ComponentType end
abstract type InputAgent <: ComponentType end
abstract type HiddenAgent <: ComponentType end
abstract type NetworkAgent <: ComponentType end

include("EcosystemStructs.jl")
include("EcosystemUtils.jl")
include("Visualise.jl")
include("Shapley.jl")

end # module
