
using Graphs, GraphRecipes
using BSON, JSON
using Plots, Colors

function simple_visu(g::AbstractGraph, c::Dict{String, Component})
    colors = Vector{Color}([])
    comps::Vector{Pair{String, Component}} = sort(collect(c), by = x -> x[2].id)
    for i=1:length(comps)
        if typeof(comps[i][2]).parameters[1] == InputAgent
            append!(colors, [colorant"red"])
        elseif typeof(comps[i][2]).parameters[1] == HiddenAgent
            append!(colors, [colorant"blue"])
        elseif typeof(comps[i][2]).parameters[1] == NetworkAgent
            append!(colors, [colorant"green"])
        end
    end
    #savefig(graphplot(g, names=map(x->x.model, comps), markercolor=colors), path)
    graphplot(g, names=map(x->x[1], comps), markercolor=colors)
end
