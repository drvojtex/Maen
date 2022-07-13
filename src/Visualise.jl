
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

#=

using Erdos

g = DiNetwork(5)
add_edge!(g, 1, 2)
add_edge!(g, 1, 3)
add_edge!(g, 3, 4)
add_edge!(g, 4, 5)
vprop!(g, "label", VertexMap(g, ["input1", "h1", "h2", "h3", "net"]));
colors = [
    :([fill "#ff2222" targetArrow "standard"]),
    :([fill "#2222ff" targetArrow "standard"]),
    :([fill "#2222ff" targetArrow "standard"]),
    :([fill "#2222ff" targetArrow "standard"]),
    :([fill "#22ff22" targetArrow "standard"]),
]
vprop!(g, "graphics", VertexMap(g, colors))
writenetwork("mygraph.gml", g)

=#
