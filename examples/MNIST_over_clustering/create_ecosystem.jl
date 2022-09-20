
using Maen 
using Flux, StatsBase, BSON
import Maen: model

@info "Creating ecosystem..."

graph, comps = create_ecosystem("setup_louvain.json");  # create the ecosystem graph and components
sch = scheduling(graph);  # find scheduling order
sch_components = scv(comps, sch);  # vector of scheduled components

params_objects = []  # list of parameters objects


function get_model_dense(in::Int64, out::Int64)
    d = Dense(in, out)
    function get_model_dense(x)
        d(x)
    end
    append!(params_objects, [d])
    return get_model_dense
end
clustering_map = BSON.load("overall_map_louvain.bson")[:map]
for (i::Int64, in::Int64) in zip(0:maximum(clustering_map)-1, 
    values(sort(countmap(clustering_map))))
    comps[string("hidden_", i)].model = get_model_dense(in, 1)
end

function get_model_net(in::Int64, out::Int64)
    d = Dense(in, out)
    function get_model_net(x)
        d(reduce(vcat, x))
    end
    append!(params_objects, [d])
    return get_model_net
end
comps["net"].model = get_model_net(maximum(clustering_map), 10)

# (key, value) = (input component id, order in sample)
inputs_ids = Dict(1:maximum(clustering_map) .=> 1:maximum(clustering_map));  

# create the ecosystem
eco = Ecosystem(graph, sch_components, inputs_ids, sch, comps, params_objects);  

function mapping(data::Any)
    map(i -> data[:,:,1][findall(x->x==i, clustering_map)], 1:maximum(clustering_map))
end

function model(data::Any)
    reduce(vcat, Maen.model(eco, data)[end])
end
