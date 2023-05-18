
using Flux, HypothesisTests, StatsBase, Statistics
using DataFrames
using Graphs, SimpleWeightedGraphs, CDalgs
using MLDatasets: BostonHousing
using Logging

@info "Dataset loading"
dataset = BostonHousing()
println("Features names: ", dataset.metadata["feature_names"], "\n")

@info "Features exploration & creating similarity (Pearson correlation) graph"
g = SimpleWeightedGraph(length(dataset.metadata["feature_names"]))
α = 0.001
for n1 in dataset.metadata["feature_names"]
    for n2 in dataset.metadata["feature_names"]
        p = pvalue(CorrelationTest(dataset.features[!,n1], dataset.features[!,n2]))
        if p < α
            add_edge!(g,
                findfirst(x -> x == n1, dataset.metadata["feature_names"]),
                findfirst(x -> x == n2, dataset.metadata["feature_names"]),
                cor(dataset.features[!,n1], dataset.features[!,n2])
            )
        end
    end
end

@info "Performing Louvain communities detection on similarity graph"
#clusters_mapping = louvain_clustering(g)
clusters_mapping = [1, 2, 1, 3, 1, 2, 1, 4, 1, 1, 1, 4, 1]
mapping_dict = Dict()
println("Communities of data features:")
for m in sort(unique(clusters_mapping))
    mapping_dict[m] = dataset.metadata["feature_names"][findall(x -> x == m, clusters_mapping)]
    println(mapping_dict[m])
end
println()

