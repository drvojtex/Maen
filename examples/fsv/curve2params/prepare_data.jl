
include("perform_clustering.jl")

dataset_a = BSON.load("../data/dataset_a.bson")
data_tensor = dataset_a[:data_tensor] * (-1);

clusters = perform_clustering(data_tensor, 19)

data_tensor[:,:,1] ./= maximum(data_tensor[:,:,1])
data_tensor[:,:,2] ./= maximum(data_tensor[:,:,2])

data = map(s -> 
    map(i -> 
        s[findall(x -> x == i, clusters), :], 1:19)
    , eachslice(data_tensor; dims=1)
)

#=
for i=1:100
    for j=12:19
        data[i][j] .*= 0
    end
end
=#

labels = deepcopy(Matrix(dataset_a[:parameters]))
for i=1:4
    labels[:,i] .-= minimum(labels[:,i])
    labels[:,i] ./= maximum(labels[:,i])
end
labels = permutedims(labels, (2, 1))

labels[findall(x->x==0, labels)] .=  minimum(filter(x->x!=0, labels))/100

shuffle_vec = shuffle(1:length(data))

function minibatch()
    s = 1:80
    deepcopy(data)[shuffle_vec[s]], deepcopy(labels)[:, shuffle_vec[s]]
end

function testbatch()
    s = 81:100
    deepcopy(data)[shuffle_vec[s]], deepcopy(labels)[:, shuffle_vec[s]]
end
