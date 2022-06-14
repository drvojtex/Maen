module Maen

export shapley, create_ecosystem
export Ecosystem
export Network, InputAgent, HiddenAgent
export include

abstract type Component end
abstract type Agent <: Component end

include("Shapley.jl")
include("EcosystemStructs.jl")
include("EcosystemUtils.jl")
include("Visualise.jl")

end # module
