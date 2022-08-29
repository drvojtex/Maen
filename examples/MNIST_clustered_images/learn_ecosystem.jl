
using Maen
using MLDatasets, StatsBase, Statistics, IterTools, MLDataPattern
using EvalMetrics, Flux
using Flux: @epochs
using JSON


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


@info "Training..."

minibatchrun = (x) -> map(d->model(d), x)

acc = (x, y) -> mean(argmax.(minibatchrun(x)).-1 .== y)

loss = (x, y) -> Flux.logitcrossentropy(reduce(hcat, minibatchrun(x)), y)

cb = () -> println("accuracy = ", acc(testbatch()[1], testbatch()[2]), 
    " loss = ", loss(testbatch()[1], Flux.onehotbatch(testbatch()[2], 0:9)))

@epochs 10 Flux.Optimise.train!(
            loss, Flux.params(eco.ps_obj), repeatedly(minibatch, 300), 
            ADAM(), cb = Flux.throttle(cb, 4)
        )

println("total train: accuracy = ", acc(data_x, data_y), " loss = ", loss(data_x, y))
println("total test: accuracy = ", acc(test_data_x, test_data_y), " loss = ", loss(test_data_x, Flux.onehotbatch(test_data_y, 0:9)))
