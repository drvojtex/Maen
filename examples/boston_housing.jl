
using Flux, IterTools
using HypothesisTests, StatsBase, Statistics, Random, LinearAlgebra
using DataFrames, Graphs, SimpleWeightedGraphs
using Maen, CDalgs
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

@info "Transforming dataset samples"
df = deepcopy(dataset.features)
mapcols!(x -> x .- minimum(x), df)
mapcols!(x -> x ./ maximum(x), df)
data = map(s -> 
    map(i ->  Vector{Float32}(collect(s[mapping_dict[i]])), 1:length(keys(mapping_dict))),
    eachrow(df)
)

@info "Preparing labels"
labels = Float32.(dataset.targets)[!, 1]
labels .-= minimum(labels)
labels ./= maximum(labels)

@info "Preparing minibatch and test batch"
shuffle_vec = shuffle(1:length(labels))
function minibatch()
    tmp_sv = shuffle(shuffle_vec[1:Int(floor(length(shuffle_vec)*0.8))])
    s = 1:Int(floor(length(tmp_sv)*0.8))
    deepcopy(data)[tmp_sv[s]], deepcopy(labels)[tmp_sv[s]]'
end
function testbatch()
    s = Int(floor(length(shuffle_vec)*0.8)):length(shuffle_vec)
    deepcopy(data)[shuffle_vec[s]], deepcopy(labels)[shuffle_vec[s]]'
end

@info "Creating neural network model"
eco = create_ecosystem("boston_housing_topology.xgml")
eco.ii = Dict( 
    map(x -> x.id, sort(filter(x->typeof(x)==Component{InputAgent}, collect(values(eco.comps))), by=x->parse(Int, replace(x.name, "in"=>""))))
    .=>
    1:length(keys(mapping_dict))
)
params_objects = []
function get_model_dense(dims)
    d1 = Dense(dims, 10, relu)
    d2 = Dense(10, 10, relu)
    function get_model_dense(x)
        d2(d1(x))
    end
    append!(params_objects, [d1, d2])
    return get_model_dense
end
for i=1:length(keys(mapping_dict))
    eco.comps[string("h",i)].model = 
        get_model_dense(length(mapping_dict[i]))
end
function get_model_out(dims)
    d = Dense(dims, 1)
    function get_model_out(x)
        d(vcat(x...))
    end
    append!(params_objects, [d])
    return get_model_out
end
eco.comps["out"].model = get_model_out(10*length(keys(mapping_dict)))
eco.schc = scv(eco.comps, eco.sch)
eco.ps_obj = params_objects
function nn(input_data::Any)
    reduce(vcat, Maen.model(eco, input_data)[end])
end

@info "Training neural network model"
batchrun = s -> reduce(hcat, map(x -> nn(x), s))
loss = (x, y) -> Flux.mse(batchrun(x), y)
rerr = (x, y) -> median(abs.(map(x->nn(x), x) .- y)./abs.(y))
R2 = (x,y) -> 1 - sum((batchrun(x).-y).^2)/sum((y.-mean(y)).^2)
cb = () -> (
    println(        
        " loss = (", loss(minibatch()...), ", ", loss(testbatch()...), " )", 
        " rerr = (", rerr(minibatch()...), ", ", rerr(testbatch()...), " )",
        " R2 = (", R2(minibatch()...), ", ", R2(testbatch()...), " )"
    )
)    
Flux.Optimise.train!(
    loss, Flux.params(eco.ps_obj), 
    repeatedly(minibatch, 1000), ADAM(),
    cb = Flux.throttle(cb, 1)
)
println("total train: relative error = ", rerr(minibatch()...), " loss = ", loss(minibatch()...));
println("total test: realtive error = ", rerr(testbatch()...), " loss = ", loss(testbatch()...));
println("R2 score: train = ", R2(minibatch()...), ", test = ", R2(testbatch()...))
println("Trainable parameters count: ", sum(length, Flux.params(eco.ps_obj)))

@info "Perform Shapley values explainability"
subset_rerr(S, x, y) = 1/median(((abs.(reduce(hcat, map(s -> subset_model(eco, s, S)[end], x)).-y))./y))
input_shaps = Dict(
    filter(x->x.id == key, collect(values(eco.comps)))[1].name =>
    value
    for (key, value) in 
    inputagents_shapley(eco, testbatch()[1], testbatch()[2], subset_rerr, monteCarlo=false)
)
hidden_shaps = Dict(
    filter(x->x.id == key, collect(values(eco.comps)))[1].name =>
    value
    for (key, value) in 
    hiddenagents_shapley(eco, testbatch()[1], testbatch()[2], subset_rerr, monteCarlo=false)
)

