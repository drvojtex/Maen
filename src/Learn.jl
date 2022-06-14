
using Flux, Zygote
using BSON, JSON
using Graphs

function load_model(path::String) 
    ecosystem::Dict{Symbol, Any} = BSON.load("$path/ecosystem.bson")
    include("$path/functions.jl")

    @show model([randn(1), randn(1)])

    map(x->x[1].=x[2], zip(model_params, ecosystem[:model_params]))

    @show model([randn(1), randn(1)])

end
