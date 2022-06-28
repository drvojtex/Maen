module Maen

export create_ecosystem, model, scv, scheduling, simple_visu
export Component, Ecosystem
export InputAgent, HiddenAgent, NetworkAgent
export shapley

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
