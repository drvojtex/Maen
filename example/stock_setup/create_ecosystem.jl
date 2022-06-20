
using Maen

include("setup.jl")

graph, components = create_ecosystem("setup.json")
sch = scheduling(graph)
sch_components = schedule_components_2_vec(components, sch)
inputs_ids = Dict(1=>1, 2=>2, 3=>3, 4=>4)

l = LSTM(3, 3)
components["hidden1"].model = l

data = [randn(Float32, 3), randn(Float32, 3), randn(Float32, 3), randn(Float32, 3)]
gs = gradient(Flux.params(l)) do
    Flux.mse(model(graph, sch_components, inputs_ids, sch, data), zeros(3))
end
@show gs.gradient
