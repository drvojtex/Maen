
function perform_clustering(data_tensor, clusters_cnt)

    function get_clusters(clustering)
        clustering(dtw_graph(data_tensor))
    end

    ncc95(g) = nc_clustering(g; α=.95)
    ncc99(g) = nc_clustering(g; α=.99)
    ncc999(g) = nc_clustering(g; α=.999)
    clustering_algs = [louvain_clustering, ncc95, ncc99, ncc999, cdep_clustering]

    clustering_alg = clustering_algs[1]

    clusters = get_clusters(clustering_alg);
    while length(unique(clusters)) != clusters_cnt
        clusters = get_clusters(clustering_alg)
    end

    println("There are ", length(unique(clusters)), " clusters.\n",
        "Create network with ", length(unique(clusters)), " InputAgents and one OutputAgent. 
        The OutputAgent should return 4 scalars correspondign to the labes of material parameters.");

    mapping = Dict(unique(clusters) .=> 1:clusters_cnt)
    new_clusters = zeros(length(clusters))
    for i::Int64 in unique(clusters)
        new_clusters[findall(x -> x==i, clusters)] .= mapping[i]
    end

    return Int.(new_clusters)
end
