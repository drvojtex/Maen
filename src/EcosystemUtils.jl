
using Graphs
using BSON, JSON
using Flux, Zygote

function create_ecosystem(sf::String)

    # Load the setup file
    sj::Dict{String, Any} = JSON.parsefile(sf)

    # Create the ecosystem graph
    g::SimpleDiGraph{Int64} = create_graph(sj)

    # Create the ecosystem components
    c::Dict{String, Component} = create_components(sj, g)

    return g, c
end

function create_graph(setup_json::Dict{String, Any})
    function get_neighbours(name::String, j::Dict{String, Any}) 
        ns::Vector{String} = j[name]["target"]
        return Vector{Tuple{Int64, Int64}}([
            (j[name]["id"], j[x]["id"]) for x in ns if haskey(j, x)
        ])
    end
    edges = Vector{Tuple{Int64, Int64}}([])
    map(x -> append!(edges, get_neighbours(x, setup_json)), collect(keys(setup_json)))
    SimpleDiGraph(Edge.(edges))
end

function create_components(setup_json::Dict{String, Any}, graph::SimpleDiGraph{Int64})
    components = Dict{String, Component}()
    for sj in setup_json
        name = sj[1]
        copmt = sj[2]
        components[name] = Component{eval(Symbol(copmt["type"]))}(
            copmt["id"], name, 
            length(graph.badjlist[copmt["id"]]) > 1 ? (x...) -> sum(x...) : identity, 0.0
        )
    end
    return components
end

# TODO: remove n^2 complexity (its only bfs)
function scheduling(g::SimpleDiGraph{Int64})
    sch = Vector{Int64}([])
    for _=1:length(g.badjlist)
        for i=1:length(g.badjlist)
            if i ∉ sch && all(map(x -> x ∈ sch, g.badjlist[i]))
                append!(sch, i)
            end
        end
    end
    return sch
end

function scv(components::Dict{String, Component}, sch::Vector{Int64})
    sort(map(x->x[2], collect(components)), by=x->x.id)[sch]
end

function model_run(eco::Ecosystem, data)
    dims = Dict{Int64, Dict{Symbol, Tuple}}()
    values = []
    for c in eco.schc
        ids = eco.g.badjlist[c.id]
        idxs = findall(x -> x ∈ ids, eco.sch)
        idxs = length(idxs) > 1 || length(idxs) == 0 ? idxs : idxs[1]
        model_input = length(idxs) > 0 ? values[idxs] : data[eco.ii[c.id]]
        model_output = c.model(model_input)
        dims[c.id] = Dict(:in => model_input |> size, :out => model_output |> size)
        values = vcat(values, [model_output])
    end
    return values, dims
end

function model(eco::Ecosystem, data)
    return model_run(eco, data)[1]
end

function agents_dims(eco::Ecosystem, data)
    return model_run(eco, data)[2]
end
