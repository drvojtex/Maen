
using Maen
using StatsBase, Graphs
using BSON, JSON
using Flux, Zygote, MLDataPattern, Statistics, IterTools
using Flux: @epochs

include("../model1/functions.jl")

ecosystem = load_model("../model1/", model_params)

function minibatch()
    idx = sample(1:length(data), 1, replace = true)
    data[idx][1], labels[idx]
end

data = [[randn(1, 3), randn(1, 3)], [randn(1, 3), randn(1, 3)], [randn(1, 3), randn(1, 3)]]
labels = [Matrix([1 0 1]), Matrix([0 0 1]), Matrix([1 0 0])]
epochs = 10

accuracy(x,y) = mean(map(xy -> round.(model(xy[1])) == xy[2], zip(x, y)))
cb = () -> println("accuracy = ", accuracy(data, labels))
loss = (x, y) -> Flux.mse(model(x), y[1])
@epochs epochs Flux.Optimise.train!(loss, model_params, repeatedly(minibatch, 1), ADAM(), cb = Flux.throttle(cb, 2))
