
using Maen 
using Flux, StatsBase, BSON
import Maen: model

@info "Creating ecosystem..."

graph, comps = create_ecosystem("setup.json");  # create the ecosystem graph and components
sch = scheduling(graph);  # find scheduling order
sch_components = scv(comps, sch);  # vector of scheduled components
inputs_ids = Dict(1:10 .=> 1:10);  # (key, value) = (input component id, order in sample)

params_objects = []  # list of parameters objects


function get_model_net()
    function get_model_net(x...)
        vcat(x...)
    end
    return get_model_net
end
comps["net"].model = get_model_net()


function get_model_dense(in::Int64, out::Int64)
    d = Dense(in, out)
    function get_model_dense(x)
        d(x)
    end
    append!(params_objects, [d])
    return get_model_dense
end

clustering_maps = BSON.load("mnist_trn_clusters.bson")[:trn_nc_mat]

for h=0:9
    comps[string("hidden_", h)].model = get_model_dense(
        length.(unique.(clustering_maps))[h+1], 1
    )
end

# create the ecosystem
eco = Ecosystem(graph, sch_components, inputs_ids, sch, comps, params_objects);  

function model(data::Any)
    tmp::Vector{Vector{Float64}} = map(idx -> median.(map(
        x -> data[findall(y -> y == x, clustering_maps[idx])], 
        unique(clustering_maps[idx])
    )), 1:10)
    reduce(vcat, Maen.model(eco, tmp)[end])
end
