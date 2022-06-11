
using JSON, Graphs

function create_ecosystem(setup_file::String)

    # Load the setup file
    setup_json::Dict{String, Any} = JSON.parsefile(setup_file)

    #
    map(xy->xy[2][2]["id"]=xy[1], enumerate(setup_json))
    #

    # Create the ecosystem graph
    graph::SimpleDiGraph{Int64} = create_graph(setup_json)

    # Create the ecosystem components
    components::Dict{String, Component} = create_components(setup_json)
    
end

function create_graph(setup_json::setup_json::Dict{String, Any})
    function get_neighbours(name::String, j::Dict{String, Any}) 
        ns::Vector{String} = j[name]["target"]
        return Vector{Tuple{Int64, Int64}}([
            (j[name]["id"], j[x]["id"]) for x in ns if haskey(j, x)
        ])
    end
    edges::Vector{Tuple{Int64, Int64}} = Vector{Tuple{Int64, Int64}}([])
    map(x -> append!(edges, get_neighbours(x, setup_json)), collect(keys(setup_json)))
    return SimpleDiGraph(edges) 
end

function create_components(setup_json::setup_json::Dict{String, Any})
    components::Dict{String, Component} = Dict{String, Component}()

    for sj in setup_json
        name = sj[1]
        component = sj[2]
        if component["type"] == "input"
            components[name] = InputAgent(
                id = component["id"],
                name = name,
                shape = Tuple(component["shapes"]["in"]),
                data = zeros(Float32, Tuple(component["shapes"]["in"]))
            )
        elseif component["type"] == "hidden"
            components[name] = HiddenAgent(
                id = component["id"],
                name = name,
                model = getfield(Main, Symbol(component["model"])),
                in_shape = Tuple(component["shapes"]["in"]),
                out_shape = Tuple(component["shapes"]["out"]),
                data = zeros(Float32, Tuple(component["shapes"]["in"]))
            )
        elseif component["type"] == "network"
            components[name] = Network(
                id = component["id"],
                name = name,
                model = getfield(Main, Symbol(component["model"])),
                in_shape = Tuple(component["shapes"]["in"]),
                out_shape = Tuple(component["shapes"]["out"]),
                data = zeros(Float32, Tuple(component["shapes"]["in"]))
            )
        end
    end
    return components
end
