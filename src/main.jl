module Maen

export shapley
export Network, Agent

abstract type Component end

include("Shapley.jl")
include("Ecosystem.jl")

end # module
