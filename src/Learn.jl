
using Flux, Zygote
using BSON, JSON
using Graphs

function load_model(path::String, ps::Zygote.Params{Zygote.Buffer{Any, Vector{Any}}}) 
    ecosystem::Dict{Symbol, Any} = BSON.load("$path/ecosystem.bson")
    map(x->x[1].=x[2], zip(ps, ecosystem[:model_params]))
    return ecosystem
end
