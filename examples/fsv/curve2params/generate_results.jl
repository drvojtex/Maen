using CDalgs, Maen
using BSON, DataFrames
using Graphs, SimpleWeightedGraphs
using Flux, StatsBase, LinearAlgebra, IterTools, Random
using CategoricalArrays

include("prepare_data.jl")
include("create_eco.jl")


function minibatch()
    s = 20:100
    deepcopy(data)[s], deepcopy(labels)[:, s]
end
function testbatch()
    s = 1:20
    deepcopy(data)[s], deepcopy(labels)[:, s]
end
include("learn_eco.jl")

tmp_result = Dict(:acc=>Vector{Tuple{Float64,Float64}}(acc_eco), :loss=>Vector{Tuple{Float64,Float64}}(loss_eco), :accp_train=>accp(minibatch()...), :accp_test=>accp(testbatch()...))
