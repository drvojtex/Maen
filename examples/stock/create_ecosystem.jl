
using Maen, Flux

println("Creating ecosystem...")

graph, components = create_ecosystem("setup.json");  # create the ecosystem graph and components
sch = scheduling(graph);  # find scheduling order
sch_components = scv(components, sch);  # vector of scheduled components
inputs_ids = Dict(3=>1, 4=>2, 1=>3, 2=>4);  # (key, value) = (input component id, order in sample)

params_objects = []  # list of parameters objects

function get_model_ccd(in::Int64, out::Int64)
    c1 = Conv((11,), 1 => 10, tanh)
    c2 = Conv((11,), 10 => 1, tanh)
    d = Dense(in-20, out)
    function get_model_ccd(x)
        x = reshape(x, (length(x)..., 1, 1))
        x = c2(c1(x)) |> Flux.flatten
        d(x)
    end
    append!(params_objects, [d, c1, c2])
    return get_model_ccd
end
in_dim = 50
hidden_out_dim = 10
components["hidden3"].model = get_model_ccd(in_dim, hidden_out_dim)
components["hidden4"].model = get_model_ccd(in_dim, hidden_out_dim)
components["hidden1"].model = get_model_ccd(in_dim, hidden_out_dim)
components["hidden2"].model = get_model_ccd(in_dim, hidden_out_dim)

function get_model_concatdense(in::Int64, out::Int64)
    d = Dense(in, out, relu6)
    function get_model_concatdense(x...)
        d(reduce(vcat, x...))
    end
    append!(params_objects, [d])
    return get_model_concatdense
end
components["hidden_dense"].model = get_model_concatdense(hidden_out_dim*4, 10)
components["net1"].model = get_model_concatdense(11, 1)

function get_model_stats()
    d = Dense(3, 1)
    function get_model_stats(x)
        d([var(x), cov(x), (x[end]-x[1])/x[end]])
    end
    append!(params_objects, [d])
    return get_model_stats
end
components["stat_features"].model = get_model_stats()

eco = Ecosystem(graph, sch_components, inputs_ids, sch);  # create the ecosystem
