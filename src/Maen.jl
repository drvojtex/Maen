
module Maen

export Component, Ecosystem
export InputAgent, HiddenAgent, NetworkAgent

export create_ecosystem, scv, scheduling
export model, subset_model
export hiddenagents_shapley, inputagents_shapley
export intergrads

abstract type AbstractComponent{T} end
abstract type ComponentType end
abstract type InputAgent <: ComponentType end
abstract type HiddenAgent <: ComponentType end
abstract type NetworkAgent <: ComponentType end

include("EcosystemStructs.jl")
include("EcosystemUtils.jl")
include("Shapley.jl")
include("Intergrads.jl")

include("../docs/EcosystemUtils.jl")
include("../docs/Intergrads.jl")
include("../docs/Shapley.jl")

end # module
