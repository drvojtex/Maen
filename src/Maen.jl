module Maen

export shapley, create_ecosystem, load_model
export Network, InputAgent, HiddenAgent

abstract type Component end
abstract type Agent <: Component end

include("Shapley.jl")
include("EcosystemStructs.jl")
include("EcosystemUtils.jl")
include("Visualise.jl")

end # module
