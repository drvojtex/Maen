
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

#shuffle_vec = shuffle(1:length(data))
shuffle_vec = [49, 22, 95, 15, 89, 20, 27, 24, 75, 60, 26, 19, 37, 82, 59, 50, 44, 79, 70, 9, 67, 69, 43, 1, 74, 99, 5, 12, 25, 16, 65, 88, 96, 91, 17, 4, 34, 48, 100, 72, 42, 56, 51, 57, 28, 68, 55, 33, 3, 52, 98, 64, 7, 62, 71, 54, 58, 87, 11, 35, 39, 94, 6, 30, 77, 63, 86, 2, 29, 47, 18, 13, 92, 80, 85, 40, 84, 76, 90, 10, 36, 32, 46, 61, 66, 23, 53, 8, 73, 45, 83, 93, 14, 81, 97, 21, 78, 31, 38, 41]


function minibatch()
    s = 1:100
    deepcopy(data)[shuffle_vec[s]], deepcopy(labels)[:, shuffle_vec[s]]
end

function testbatch()
    s = 1:100
    deepcopy(data)[shuffle_vec[s]], deepcopy(labels)[:, shuffle_vec[s]]
end

