module Maen

export shapley
export Ecosystem
export Network, InputAgent, HiddenAgent

abstract type Component end
abstract type Agent <: Component end

include("Shapley.jl")
include("EcosystemStructs.jl")
include("EcosystemUtils.jl")

end # module
