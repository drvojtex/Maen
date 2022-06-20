module Maen

export shapley
export create_ecosystem, model, scv, scheduling, simple_visu
export Component, Ecosystem
export InputAgent, HiddenAgent, NetworkAgent

abstract type AbstractComponent{T} end
abstract type ComponentType end
abstract type InputAgent <: ComponentType end
abstract type HiddenAgent <: ComponentType end
abstract type NetworkAgent <: ComponentType end

include("Shapley.jl")
include("EcosystemStructs.jl")
include("EcosystemUtils.jl")
include("Visualise.jl")

end # module
