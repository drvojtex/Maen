
using Maen
using Flux

graph, components = create_ecosystem("setup.json")
sch = scheduling(graph)
sch_components = scv(components, sch)
inputs_ids = Dict(1=>1, 2=>2, 3=>3, 4=>4)

l = LSTM(3, 3)
components["hidden1"].model = l

eco = Ecosystem(graph, sch_components, inputs_ids, sch)

data = [randn(Float32, 3), randn(Float32, 3), randn(Float32, 3), randn(Float32, 3)]
gs = gradient(Flux.params(l)) do
    Flux.mse(model(eco, data), zeros(3))
end
@show gs.grads
