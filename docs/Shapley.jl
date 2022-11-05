
using DocumentFunction

@doc """
$(DocumentFunction.documentfunction(shapley;
    location=false,
    maintext="
    The Shapley algorithm (solution concept in cooperative game theory) for computing 
    the Shapley values of a set of agents. The Shapley values are the expected 
    utility of each agent in a set.

    julia> shapley(
        eco, data, labels, ids, (x, S) -> argmax(
        subset_model(eco, x, S, noise=true)[end]) - 1
    )
    ",
    argtext=Dict("eco"=>"neural network ecosystem",
                 "data"=>"input data of the network",
                 "labels"=>"labels corresponding to the data",
                 "ids"=>"identifiers of components to be computed shapley values",
                 "s_model"=>"subset model with defined output format")))
""" shapley

@doc """
$(DocumentFunction.documentfunction(hiddenagents_shapley;
    location=false,
    maintext="
    Compute Shapley values of hiddent agents components.

    julia> hiddenagents_shapley(eco, data, labels, 
        (eco, x, S) -> argmax(subset_model(eco, x, S, noise=true)[end]) - 1
    )
    ",
    argtext=Dict("eco"=>"neural network ecosystem",
                 "data"=>"input data of the network",
                 "labels"=>"labels corresponding to the data",
                 "s_model"=>"subset model with defined output format")))
""" hiddenagents_shapley

@doc """
$(DocumentFunction.documentfunction(inputagents_shapley;
    location=false,
    maintext="
    Compute Shapley values of input agents components.

    julia> hiddenagents_shapley(eco, data, labels, 
        (eco, x, S) -> argmax(subset_model(eco, x, S, noise=true)[end]) - 1
    )
    ",
    argtext=Dict("eco"=>"neural network ecosystem",
                 "data"=>"input data of the network",
                 "labels"=>"labels corresponding to the data",
                 "s_model"=>"subset model with defined output format")))
""" inputagents_shapley
