module Maen

export shapley
export create_ecosystem, model, schedule_components_2_vec, schelduding
export simple_visu
export Component
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
