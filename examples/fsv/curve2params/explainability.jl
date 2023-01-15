
#using Plots

subset_acc(S, x, y) = 1/median(((abs.(reduce(hcat, map(s -> subset_model(eco, s, S)[end], x)).-y))./y))
shaps = inputagents_shapley(eco, data, labels, subset_acc, monteCarlo=true)

coms_shaps = Dict(eco.ii[key] => value for (key, value) in shaps)
cluster_shaps = []
for i=1:length(clusters)
    append!(cluster_shaps, coms_shaps[clusters[i]])
end


p = [
    histogram(map(x -> x[2], argmax(data_tensor[:,:,2], dims=2)[:,1]), labels="peaks", xlabel="agents"),
    plot(1:61, cluster_shaps, labels="shapley", xlabel="agents")
]

plot(p..., layout=(2, 1))

