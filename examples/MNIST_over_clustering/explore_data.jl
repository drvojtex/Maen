
using Maen
using MLDatasets, MLDataPattern
using StatsBase, Statistics, IterTools
using BSON, Plots


clustering_maps = BSON.load("mnist_trn_clusters.bson")[:trn_nc_mat];

#data_x = map(x -> mapping(x), eachslice(MNIST(:train).features, dims=3));
#data_y = MNIST(:train).targets;

function explore_mapping(cmaps::Vector, l::Int64, psobj::Vector{Any})

    cmap::Matrix = cmaps[l]
    Θ = Flux.params(psobj[l])[1]

    img::Vector{Float64} = vec(deepcopy(cmap))
    for id::Int64=1:maximum(cmap)
        tmp::Float64 = abs(Θ[id]) > median(abs.(Θ)) ? abs(Θ[id]) : 0
        img[findall(x->x==id, vec(cmap))] .= tmp
    end
    return reshape(img, (28, 28))
end

heatmap(explore_mapping(clustering_maps, 1, params_objects), color=:greys)
