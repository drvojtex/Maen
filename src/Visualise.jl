
using Graphs, GraphRecipes
using BSON, JSON
using Plots

function simple_visu(g::AbstractGraph, c::Vector{Component}, path::String)
    colors = Vector{Color}([])
    comps::Vector{Component} = sort(collect(values(c)), by = x -> x.id)
    for i=1:length(comps)
        if typeof(comps[i]) == InputAgent
            append!(colors, [colorant"red"])
        elseif typeof(comps[i]) == HiddenAgent
            append!(colors, [colorant"blue"])
        elseif typeof(comps[i]) == Network
            append!(colors, [colorant"green"])
        end
    end
    savefig(graphplot(g, names=map(x->x.model, comps), markercolor=colors), path)
end
