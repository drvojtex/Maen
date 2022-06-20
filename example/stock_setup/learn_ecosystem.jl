
using Maen
using StatsBase, Statistics, IterTools, Graphs
using BSON, JSON, CSV, DataFrames
using Flux, Zygote, MLDataPattern
using Flux: @epochs

include("../stock_model/functions.jl")

#ecosystem = load_model("../stock_model/", model_params)
#map(x->x[1].=x[2], zip(model_params, ecosystem[:model_params]))

println("Creating dataset...")
iterations = 100
window = 50
data = []
labels = []
df = filter(x->x.Name=="AAPL", CSV.read("all_stocks_5yr.csv", DataFrame))

for i=1:size(df)[1]-100
    a_o = randn(Float32, window)
    a_h = randn(Float32, window)
    a_l = randn(Float32, window)
    a_c = randn(Float32, window)
    b = Matrix{Float32}(zeros(1)')
    
    tmp = deepcopy(df[i:i+window-1, 2:5])  # get 100 days of stock

    for i=1:4
        tmp[:, i] .- mean(tmp[:, i])
    end
    next7 = mean(deepcopy(df[i+window-1+10:i+window-1+20, 5])) # 10-20 days later mean
    next7 .- mean(tmp[:, 4])

    a_o[:] = tmp[:, 1]
    a_h[:] = tmp[:, 2]
    a_l[:] = tmp[:, 3]
    a_c[:] = tmp[:, 4]
    
    b[1] = a_c[end] < next7 ? 1 : 0
    
    append!(data, [[a_o, a_h, a_l, a_c]])
    append!(labels, [b])
end

function minibatch()
    idx = sample(1:length(labels), 100, replace=true)
    permutedims(mapreduce(permutedims, vcat, data[idx]), (2, 1)), labels[idx]
end

println("Training...")

function runminibatch(x, y)
    mapreduce(xy->Flux.mse(model(xy[1]), xy[2]), +, zip(eachcol(x), y))
end

#accuracy(x,y) = mean(map(xy -> abs(model(xy[1])[1] .- xy[2][1])/xy[2][1] < 0.1, zip(x, y)))
accuracy(x,y) = mean(map(x->round(model(x[1])[1]) == x[2][1], zip(x, y)))
cb = () -> println("accuracy = ", accuracy(data, labels), " loss = ", loss_cb(data, labels))
loss_cb = (x, y) -> mapreduce(xy-> Flux.mse(model(xy[1]), y[2]), +, zip(x, y))
loss = (x, y) -> runminibatch(x, y)

@epochs iterations Flux.Optimise.train!(loss, model_params, repeatedly(minibatch, 100), ADAM(0.0001), cb = Flux.throttle(cb, 1))

println("total: accuracy = ", accuracy(data, labels), " loss = ", loss_cb(data, labels))
