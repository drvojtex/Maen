
using CSV, DataFrames
using StatsBase, Distributions

println("Creating dataset...")

window = 50
df = filter(x->x.Name=="JPM", CSV.read("all_stocks_5yr.csv", DataFrame))

data = []
labels = []
p = rand(Bernoulli(0.8), size(df)[1]-100)

for i=1:size(df)[1]-100
    tmp = deepcopy(df[i:i+window-1, 2:5])  # get 100 days of stock
    for j=1:4
        tmp[:, j] .-= minimum(tmp[:, j])
    end
    next7 = mean(deepcopy(df[i+window-1+10:i+window-1+20, 5])) # 10-20 days later mean
    
    append!(data, [map(x-> Vector{Float32}(x), [tmp[:, 1], tmp[:, 2], tmp[:, 3], tmp[:, 4]])])
    append!(labels, [Matrix{Float32}([Int(df[i:i+window-1, 4][end] < next7)]')])
end

data = permutedims(mapreduce(permutedims, vcat, data))
labels = reduce(vcat, labels)

function minibatch()
    idx = sample(1:length(labels), 50, replace=true)
    filter!(x -> Bool(p[x]), idx)
    data[:, idx], labels[idx]
end

function testbatch()
    idx = filter(x -> !Bool(p[x]), collect(1:size(df)[1]-100))
    data[:, idx], labels[idx]
end
