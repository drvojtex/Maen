
using Graphs
using BSON, JSON
using Flux, Zygote

function create_ecosystem(setup_file::String, functions::String, 
        ps::Zygote.Params{Zygote.Buffer{Any, Vector{Any}}}, model_path::String)

    # Load the setup file
    setup_json::Dict{String, Any} = JSON.parsefile(setup_file)

    # Create the ecosystem graph
    graph::SimpleDiGraph{Int64} = create_graph(setup_json)

    # Create the ecosystem components
    components::Dict{String, Component} = create_components(setup_json)
    
    # Create the ecosystem model
    m = get_complex_function(
        filter(x->thetype(typeof(x)) == Network, collect(values(components)))[1].id, 
        deepcopy(graph.badjlist), collect(values(components))
    )

    # Create the ecosystem folder
    mkdir(model_path)
    run(`cp $functions $model_path/functions.jl`)
    open("$model_path/functions.jl", "a") do f
        write(f, "\n$m")
    end
    bson("$model_path/ecosystem.bson", graph=graph, components=components, model_params=ps)

end

function create_graph(setup_json::Dict{String, Any})
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

function create_components(setup_json::Dict{String, Any})
    components::Dict{String, Component} = Dict{String, Component}()

    for sj in setup_json
        name = sj[1]
        component = sj[2]
        ctype = lowercase(component["type"])
        if ctype == "input"
            components[name] = InputAgent(component["id"], name)
        elseif ctype in ["hidden", "network"]
            components[name] = HiddenAgent(
                id = component["id"],
                name = name,
                model = String(Symbol(component["model"]))
            )
            if ctype == "network"
                components[name] = Network(components[name])
            end
        end
    end
    return components
end

function get_complex_function(top::Int64, badjlist::Vector{Vector{Int64}}, 
        components::AbstractArray{Component})
    map(x->append!(x, -1), badjlist)

    s = "model(x) = "
    b = false
    function dfs_rec(v)
        if v == -1
            b = true
            s *= ")"
            return
        end
        if b s *= ", " end
        b = false
        s *= "$(filter(x->x.id == v, components)[1].model)("
        if badjlist[v][1] == -1 s *= "x[$(components[v].input_id)]" end
        for w in badjlist[v] dfs_rec(w) end
    end
    dfs_rec(top)
    println(s)
    return s
end
