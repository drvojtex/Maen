# Maen - Multiple agents ecosystem network

## Overview
Maen is a framework for generating neural networks. The model is generated on topologies given in XGML format (for example, using [yEd](https://www.yworks.com/products/yed#yed-support-resources "yEd software homepage") software). The framework also includes tools for explainability of model components using [Shapley values](https://www.google.com/search?client=safari&rls=en&q=shapley+numbers+loyd&ie=UTF-8&oe=UTF-8 "Shapley values Wikipedia").

## Example of usage

### Introduction

- There is an simple example of usage on [Boston housing dataset](https://docs.juliahub.com/MLDatasets/9CUQK/0.5.13/datasets/BostonHousing/ "MLDatasets package Boston Housing"). 
- The dataset contains 506 samples, each of them has got 13 continuous attributes (including target attribute "MEDV"), 1 binary-valued attribute. 
- The goal is to perform a regression of the (continious) target variable.
- Sources: (a) Origin: This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University. (b) Creator: Harrison, D. and Rubinfeld, D.L. 'Hedonic prices and the demand for clean air', J. Environ. Economics & Management, vol.5, 81-102, 1978. (c) Date: July 7, 1993

Features of the dataset:
Symbol | Caption
-------|--------
CRIM | per capita crime rate by town
ZN | proportion of residential land zoned for lots over 25,000 sq.ft.
INDUS | proportion of non-retail business acres per town
CHAS | Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
NOX | nitric oxides concentration (parts per 10 million)
RM | average number of rooms per dwelling
AGE | proportion of owner-occupied units built prior to 1940
DIS | weighted distances to five Boston employment centres
RAD | index of accessibility to radial highways
TAX | full-value property-tax rate per 10,000 dollars
PTRATIO | pupil-teacher ratio by town
B | 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
LSTAT | % lower status of the population

Target of the dataset:
Symbol | Caption
-------|---------
MEDV | Median value of owner-occupied homes in 1000's of dollars

### Codes & commands

The following commands are in also in **exmaples** folder. 

Load libraries.
```julia
julia> using Flux, IterTools
       using HypothesisTests, StatsBase, Statistics, Random, LinearAlgebra
       using DataFrames, Graphs, SimpleWeightedGraphs
       using Maen, CDalgs
       using MLDatasets: BostonHousing
       using Logging
```

Load dataset and print features & target names.
```julia
julia> dataset = BostonHousing()
julia> println("Features names: ", dataset.metadata["feature_names"], "\n")
julia> println("Target names: ", dataset.metadata["target_names"], "\n")
```

Features exploration & creating similarity (Pearson correlation) graph.
```julia
julia> g = SimpleWeightedGraph(length(dataset.metadata["feature_names"]))
julia> α = 0.001
julia> for n1 in dataset.metadata["feature_names"]
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
```

Performing Louvain communities detection on similarity graph. The implementation of the Louvain method is from the [CDalgs library](https://github.com/drvojtex/CDalgs "CDalgs"). 
```julia
julia> clusters_mapping = louvain_clustering(g) # [1, 2, 1, 3, 1, 2, 1, 4, 1, 1, 1, 4, 1]
julia> mapping_dict = Dict()
julia> println("Communities of data features:")
julia> for m in sort(unique(clusters_mapping))
           mapping_dict[m] = dataset.metadata["feature_names"][findall(x -> x == m, clusters_mapping)]
           println(mapping_dict[m])
        end
```

Transforming dataset samples (normalisation).
```julia
julia> df = deepcopy(dataset.features)
julai> mapcols!(x -> x .- minimum(x), df)
       mapcols!(x -> x ./ maximum(x), df)
julia> data = map(s -> 
            map(i ->  Vector{Float32}(collect(s[mapping_dict[i]])),  1:length(keys(mapping_dict))),
            eachrow(df)
        )
julia> labels = Float32.(dataset.targets)[!, 1]
       labels .-= minimum(labels)
       labels ./= maximum(labels)
```

Creating minibatch and testbatch.
```julia
julia> shuffle_vec = shuffle(1:length(labels))
julia> function minibatch()
           tmp_sv = shuffle(shuffle_vec[1:Int(floor(length(shuffle_vec)*0.8))])
           s = 1:Int(floor(length(tmp_sv)*0.8))
           deepcopy(data)[tmp_sv[s]], deepcopy(labels)[tmp_sv[s]]'
       end
julia> function testbatch()
          s = Int(floor(length(shuffle_vec)*0.8)):length(shuffle_vec)
          deepcopy(data)[shuffle_vec[s]], deepcopy(labels)[shuffle_vec[s]]'
       end
```

Creating neural network model. In the file **boston_housing_topology.xgml** is defined topology of the neural network model (visible in following Figure).
![Alt text](examples/boston_housing_topology.png?raw=true) 
```julia
julia> eco = create_ecosystem("boston_housing_topology.xgml")
julia> eco.ii = Dict( 
          map(x -> x.id, sort(filter(x->typeof(x)==Component{InputAgent}, collect(values(eco.comps))), by=x->parse(Int, replace(x.name, "in"=>""))))
        .=>
          1:length(keys(mapping_dict))
       )
julia> params_objects = []
julia> function get_model_dense(dims)
          d1 = Dense(dims, 10, relu)
          d2 = Dense(10, 10, relu)
          function get_model_dense(x)
             d2(d1(x))
          end
          append!(params_objects, [d1, d2])
          return get_model_dense
       end
julia> for i=1:length(keys(mapping_dict))
          eco.comps[string("h",i)].model = 
            get_model_dense(length(mapping_dict[i]))
       end
julia> function get_model_out(dims)
          d = Dense(dims, 1)
          function get_model_out(x)
             d(vcat(x...))
          end
          append!(params_objects, [d])
          return get_model_out
       end
julia> eco.comps["out"].model = get_model_out(10*length(keys(mapping_dict)))
       eco.schc = scv(eco.comps, eco.sch)
       eco.ps_obj = params_objects
julia> function nn(input_data::Any)
          reduce(vcat, Maen.model(eco, input_data)[end])
       end
```

Training neural network model.
```julia
julia> batchrun = s -> reduce(hcat, map(x -> nn(x), s))
       loss = (x, y) -> Flux.mse(batchrun(x), y)
       rerr = (x, y) -> median(abs.(map(x->nn(x), x) .- y)./abs.(y))
       R2 = (x,y) -> 1 - sum((batchrun(x).-y).^2)/sum((y.-mean(y)).^2)
julia> cb = () -> (
          println(        
             " loss = (", loss(minibatch()...), ", ", loss(testbatch()...), " )", 
             " rerr = (", rerr(minibatch()...), ", ", rerr(testbatch()...), " )",
             " R2 = (", R2(minibatch()...), ", ", R2(testbatch()...), " )"
          )
       )    
julia> Flux.Optimise.train!(
          loss, Flux.params(eco.ps_obj), 
          repeatedly(minibatch, 100), ADAM(),
          cb = Flux.throttle(cb, 1)
       )
julia> println("total train: relative error = ", acc(minibatch()...), " loss = ", loss(minibatch()...));
       println("total test: realtive error = ", acc(testbatch()...), " loss = ", loss(testbatch()...));
       println("R2 score: train = ", R2(minibatch()...), ", test = ", R2(testbatch()...))
       println("Trainable parameters count: ", sum(length, Flux.params(eco.ps_obj)))
```

Performing explainability of components via Shapley values.
```julia
julia> subset_rerr(S, x, y) = 1/median(((abs.(reduce(hcat, map(s -> subset_model(eco, s, S)[end], x)).-y))./y))
julia> input_shaps = Dict(
          filter(x->x.id == key, collect(values(eco.comps)))[1].name => value
          for (key, value) in 
          inputagents_shapley(eco, testbatch()[1], testbatch()[2], subset_rerr, monteCarlo=false)
       )
julia> hidden_shaps = Dict(
          filter(x->x.id == key, collect(values(eco.comps)))[1].name => value
          for (key, value) in 
          hiddenagents_shapley(eco, testbatch()[1], testbatch()[2], subset_rerr, monteCarlo=false)
       )
```

## License

GNU GENERAL PUBLIC LICENSE Version 2, June 1991
