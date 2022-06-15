
using Maen
using StatsBase, Statistics, IterTools, Graphs
using BSON, JSON, CSV, DataFrames
using Flux, Zygote, MLDataPattern
using Flux: @epochs

include("../stock_model/functions.jl")

ecosystem = load_model("../stock_model/", model_params)
#model_params = ecosystem[:model_params]

function minibatch()
    idx = sample(1:length(data), 1, replace = true)
    data[idx][1], labels[idx]
end

epochs = 100
bs = 3
data = []
labels = []
df = filter(x->x.Name=="AAPL", CSV.read("all_stocks_5yr.csv", DataFrame))
for i=1:bs:size(df)[1]-200
    a_o = randn(Float32, 100, bs)
    a_h = randn(Float32, 100, bs)
    a_l = randn(Float32, 100, bs)
    a_c = randn(Float32, 100, bs)
    b = Matrix{Float32}(zeros(bs)')
    for j=1:bs
        tmp = deepcopy(df[i+(j-1):i+99+(j-1), 2:5])  # get 100 days of stock
        a_o[:, j] = tmp[:, 1]
        a_h[:, j] = tmp[:, 2]
        a_l[:, j] = tmp[:, 3]
        a_c[:, j] = tmp[:, 4]
        tmp7 = df[i+99+7+(j-1), 5] > a_c[end]*1.1   # 7 days later
        b[j] = tmp7
    end
    append!(data, [[a_o, a_h, a_l, a_c]])
    append!(labels, [b])
end

accuracy(x,y) = mean(map(xy -> round.(model(xy[1])) == xy[2], zip(x, y)))
cb = () -> println("accuracy = ", accuracy(data, labels), " loss = ", loss_cb(data, labels))
loss_cb = (x, y) -> mapreduce(xy-> Flux.mse(model(xy[1]), y[2]), +, zip(x, y))
loss = (x, y) -> Flux.mse(model(x), y[1])
@epochs epochs Flux.Optimise.train!(loss, model_params, repeatedly(minibatch, 1), ADAM(), cb = Flux.throttle(cb, 2))
