
using BSON, DataFrames
using StatsBase, LinearAlgebra, IterTools, Random
using Flux


dataset_a = BSON.load("data/dataset_a.bson")

#size(data) = 100×61×2
data_tensor = dataset_a[:data_tensor] * (-1);
data_tensor[:,:,1] ./= maximum(data_tensor[:,:,1])
data_tensor[:,:,2] ./= maximum(data_tensor[:,:,2])
data_tensor = permutedims(data_tensor, (2,3,1))

labels = deepcopy(Matrix(dataset_a[:parameters]))
for i=1:4
    labels[:,i] .-= minimum(labels[:,i])
    labels[:,i] ./= maximum(labels[:,i])
end
labels = permutedims(labels, (2,1))

shuffle_vec = shuffle(1:100)

function minibatch()
    deepcopy(data_tensor)[:,:,shuffle_vec[1:80]], deepcopy(labels)[:,shuffle_vec[1:80]]
end

function testbatch()
    deepcopy(data_tensor)[:,:,shuffle_vec[80:end]], deepcopy(labels)[:,shuffle_vec[80:end]]
end

m = Chain(x -> reshape(x, (size(x)...,1)), Flux.flatten, Dense(122, 1000), Dense(1000, 500), Dense(500, 160), Dense(160, 40), Dense(40, 4))
ps = Flux.params(m)

batchrun = x -> reduce(hcat, map(z -> m(z), eachslice(x, dims=3)))
loss = (x, y) -> Flux.mse(batchrun(x), y)
acc = (x, y) -> Flux.mae(batchrun(x), y)

cb = () -> println("acc = (", acc(minibatch()...), ", ", acc(testbatch()...), " )",
    " loss = (", loss(minibatch()...), ", ", loss(testbatch()...), " )")

Flux.Optimise.train!(
    loss, ps, 
    repeatedly(minibatch, 100), ADAM(),
    cb = Flux.throttle(cb, 2)
)


