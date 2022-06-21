
using CSV, DataFrames
using StatsBase

println("Creating dataset...")

window = 50
df = filter(x->x.Name=="AAPL", CSV.read("all_stocks_5yr.csv", DataFrame))

data = []
labels = []

for i=1:size(df)[1]-100
    tmp = deepcopy(df[i:i+window-1, 2:5])  # get 100 days of stock
    for i=1:4
        tmp[:, i] .- mean(tmp[:, i])
    end
    next7 = mean(deepcopy(df[i+window-1+10:i+window-1+20, 5])) # 10-20 days later mean
    next7 .- mean(tmp[:, 4])
    
    append!(data, [map(x-> Vector{Float32}(x), [tmp[:, 1], tmp[:, 2], tmp[:, 3], tmp[:, 4]])])
    append!(labels, [Matrix{Float32}([Int(tmp[:, 4][end] < next7)]')])
end

data = permutedims(mapreduce(permutedims, vcat, data))

function minibatch()
    idx = sample(1:length(labels), 10, replace=true)
    data[:, idx], labels[idx]
end
