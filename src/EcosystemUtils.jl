
using JSON, Graphs

function create_ecosystem(setup_file::String)

    # Load the setup file
    setup_json::Dict{String, Any} = JSON.parsefile(setup_file)

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
    return SimpleDiGraph(Edge.(edges))
end

function create_components(setup_json::setup_json::Dict{String, Any})
    components::Dict{String, Component} = Dict{String, Component}()

    for sj in setup_json
        name = sj[1]
        component = sj[2]
        ctype = lowercase(component["type"])
        if ctype == "input"
            components[name] = InputAgent(
                id = component["id"],
                name = name
            )
        elseif ctype in ["hidden", "network"]
            components[name] = HiddenAgent(
                id = component["id"],
                name = name,
                model = getfield(Main, Symbol(component["model"]))
            )
            if ctype == "network"
                components[name] = Network(components[name])
            end
        end
    end
    return components
end

function connect_components(c_id::Int64, cin_ids::AbstractArray{Inf64},
                           components::Dict{String, Component})
    c::Component = filter(x->x.id == c_id, components)[1]

end
