
using Maen
using StatsBase

include("../model1/functions.jl")

ecosystem = load_model("../model1/", model_params)

function minibatch()
    idx = sample(1:length(data), minibatchsize, replace = true)
    data[idx], labels[idx]
end

data = [randn(1), randn(1), randn(1), randn(1), randn(1), randn(1), randn(1), randn(1)]
labels = [1, 0, 1, 0, 0, 1, 1, 0]
minibatchsize = 3

accuracy(x,y) = mean(map(xy -> round.(model(xy[1])) == xy[2], zip(x, y)))
cb = () -> println("accuracy = ", accuracy(data, labels))
loss = (x, y) -> Flux.mse(model(x), y)
Flux.Optimise.train!(loss, model_params, repeatedly(minibatch, 10), ADAM(), cb = Flux.throttle(cb, 2))
