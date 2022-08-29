
using Maen 
using Flux, StatsBase, BSON
import Maen: model

@info "Creating ecosystem..."

graph, comps = create_ecosystem("setup.json");  # create the ecosystem graph and components
sch = scheduling(graph);  # find scheduling order
sch_components = scv(comps, sch);  # vector of scheduled components
inputs_ids = Dict(1 => 1);  # (key, value) = (input component id, order in sample)

params_objects = []  # list of parameters objects


comps["net"].model = identity

function get_model_dense(in::Int64, out::Int64)
    d1 = Dense(in, 20)
    d2 = Dense(20, 20)
    d3 = Dense(20, out)
    function get_model_dense(x)
        d3(d2(d1(x)))
    end
    append!(params_objects, [d1, d2, d3])
    return get_model_dense
end

comps["hidden"].model = get_model_dense(28*28, 10)


# create the ecosystem
eco = Ecosystem(graph, sch_components, inputs_ids, sch, comps, params_objects);  

#=
clustering_maps = BSON.load("mnist_trn_clusters.bson")[:trn_nc_mat]
function mapping(data::Any)
    map(idx -> mean.(map(
        x -> data[findall(y -> y == x, clustering_maps[idx])], 
        unique(clustering_maps[idx])
    )), 1:10)
end
=#

function model(data::Any)
    reduce(vcat, Maen.model(eco, [data])[end])
end
