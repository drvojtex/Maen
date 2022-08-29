
using BSON, Plots

clustering_maps = BSON.load("mnist_trn_clusters.bson")[:trn_nc_mat]

function explore_mapping(cmap::Matrix, data::Any, labels::Any, l::Int64)
    minibatchrun = (x) -> map(d->model(d), eachrow(x))
    
    correct_data = data[findall(b -> b == 1, argmax.(minibatchrun(data)).-1 .== labels), :]
    correct_labels = vec(labels[findall(b -> b == 1, argmax.(minibatchrun(data)).-1 .== labels), :])[:,1]
    
    correct_data = correct_data[findall(lb -> lb == l, correct_labels), :]
    correct_labels = correct_labels[findall(lb -> lb == l, correct_labels), :]

    acc = (x, y) -> mean(argmax.(minibatchrun(x)).-1 .== y)

    cvec = []
    for c::Int64 in unique(cmap)
        tmp_data = deepcopy(correct_data)
        tmp_data[:, findall(x -> x == c, vec(cmap))] .= 0
        #@show c, acc(correct_data, correct_labels), acc(tmp_data, correct_labels)
        append!(cvec, acc(correct_data, correct_labels) - acc(tmp_data, correct_labels))
    end
    cvec .-= minimum(cvec)
    cvec ./= maximum(cvec)

    cvec[findall(x -> x <= median(unique(cvec)), cvec)] .= 0

    img::Vector{Float64} = vec(deepcopy(cmap))
    for i::Int64=1:length(cvec)
        img[findall(x->x==i, vec(cmap))] .= cvec[i]
    end
    return reshape(img, (28, 28))
end

heatmap(explore_mapping(clustering_maps[3], data_x, data_y, 3), color=:greys)
