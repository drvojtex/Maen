
using Graphs
using BSON, JSON
using Flux, Zygote


function create_ecosystem(xgml::String)
    # Parse the setup xgml to the Network
    net::Network = xgml2network(xgml)
    # Create the ecosystem graph
    g::SimpleDiGraph{Int64} = create_graph(net)
    # Create the ecosystem components
    comps::Dict{String, Component} = create_components(net, g)
    # Find the scheduling order
    sch = scheduling(g)
    # Vector of scheduled components
    sch_components = scv(comps, sch)

    return Ecosystem(g, sch_components, Dict(), sch, comps, Flux.params([]))
end

function create_graph(net::Network)
    g::SimpleDiGraph = SimpleDiGraph(length(net.ids))
    for s::Int64 in keys(net.adjacency_list)
        map(t::Int64 -> add_edge!(g, s, t), net.adjacency_list[s])
    end
    return g
end

function create_components(net::Network, graph::SimpleDiGraph{Int64})
    components = Dict{String, Component}()
    for id::Int64 in net.ids
        type::String = net.types[id]
        name::String = net.names[id]
        components[name] = Component{eval(Symbol(type))}(
            id, name, 
            length(graph.badjlist[id]) > 1 ? (x...) -> sum(x...) : identity, 0.0
        )
    end
    return components
end

function scheduling(g::SimpleDiGraph{Int64})
    sch = Vector{Int64}([])
    nonassigned = Set{Int64}(1:length(Graphs.vertices(g)))
    while !isempty(nonassigned)
        i::Int64 = first(filter(
            顶点1 -> 顶点1 ∉ sch && 
            all(map(顶点2 -> 顶点2 ∈ sch, g.badjlist[顶点1]))
        , nonassigned))
        append!(sch, i)
        setdiff!(nonassigned, i)
    end
    return sch
end

function scv(components::Dict{String, Component}, sch::Vector{Int64})
    sort(map(零件 -> 零件[2], collect(components)), by = 零件 -> 零件.id)[sch]
end

function model_output(eco::Ecosystem, c::T, data::Any, values::Any) where T <: Component
    ids = eco.g.badjlist[c.id]
    idxs = findall(x -> x ∈ ids, eco.sch)
    idxs = length(idxs) > 1 || length(idxs) == 0 ? idxs : idxs[1]
    model_input = length(idxs) > 0 ? values[idxs] : data[eco.ii[c.id]]
    c.model(model_input)
end

function model(eco::Ecosystem, data::Any)
    values = []
    for c::Component in eco.schc
        values = vcat(values, [
            model_output(eco, c, data, values)
        ])
    end
    return values
end

function subset_model(eco::Ecosystem, data::Any, subset::Vector{Int64}; noise::Bool=false)
    values = []
    for c::Component in eco.schc
        mo = model_output(eco, c, data, values)
        mo = c.id ∈ subset ? mo : (
            noise ? map(tmp -> tmp .* randn(Float32), mo) : map(tmp -> tmp .* Float32(.0), mo)
        )
        values = vcat(values, [mo])
    end
    return values
end
