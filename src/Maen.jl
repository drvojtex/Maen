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

function include(s::String)
    if isdir(s)
        for f in glob(s + "/*.jl")
            include(f)
        end
    else
        include(s)
    end
end

end # module
