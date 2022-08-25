
module Maen

export Component, Ecosystem
export InputAgent, HiddenAgent, NetworkAgent

export create_ecosystem, scv, scheduling
export model, subset_model
export hiddenagents_shapley, inputagents_shapley, cluster_relations
export intergrads
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
include("Intergrads.jl")

end # module
