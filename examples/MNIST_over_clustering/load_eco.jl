
using Maen
using MLDatasets, StatsBase, Statistics, IterTools, MLDataPattern
using EvalMetrics, Flux
using Flux: @epochs
using JSON, BSON
using Graphs, SimpleWeightedGraphs

include("create_ecosystem.jl")

@info "Preparing data..."

data_x = map(x -> mapping(x), eachslice(MNIST(:train).features, dims=3));
data_y = MNIST(:train).targets;
y = Flux.onehotbatch(data_y, 0:9);
function minibatch()
    idx = sample(1:length(data_y), 500, replace=true)
    data_x[idx, :], Flux.onehotbatch(data_y[idx], 0:9)
end

test_data_x = map(x -> mapping(x), eachslice(MNIST(:test).features, dims=3));
test_data_y = MNIST(:test).targets;
function testbatch()
    test_data_x, test_data_y
end

minibatchrun = (x) -> map(d->model(d), x)
acc = (x, y) -> mean(argmax.(minibatchrun(x)).-1 .== y)

eco = BSON.load("trn_eco_louvain.bson")[:eco];

