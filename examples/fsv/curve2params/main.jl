using CDalgs, Maen
using BSON, DataFrames
using Graphs, SimpleWeightedGraphs
using Flux, StatsBase, LinearAlgebra, IterTools, Random
using CategoricalArrays

include("prepare_data.jl")
include("create_eco.jl")
include("learn_eco.jl")
