
using Maen
using Graphs, SimpleWeightedGraphs
using Flux, StatsBase, LinearAlgebra


# Load xgml file and setup inputs mapping
eco = create_ecosystem("../data/network.xgml");
eco.ii = Dict( 
    map(x -> x.id, sort(filter(x->typeof(x)==Component{InputAgent}, collect(values(eco.comps))), by=x->parse(Int, replace(x.name, "in"=>""))))
    .=>
    1:19
);

# Vector of trainable objects
params_objects = []


########################################################
# Rotations

function get_model_rotations_r_I(dims::Int64)
    d = Dense(dims, dims; init = (a, b) -> randn(ComplexF32, a, b))
    function get_model_rotations_r_I(x)
        d(x[:,1] .+ x[:,2]*im)
    end
    append!(params_objects, [d])
    return get_model_rotations_r_I
end
for i=1:19
    eco.comps[string("r_I_",i)].model = 
        get_model_rotations_r_I(length(filter(x -> x == i, clusters)))
end


########################################################
# Laurent polynom
function get_model_rotations_r_II(in::Int64, out::Int64)
    d = Dense(in, out; init = (a, b) -> randn(ComplexF32, a, b))
    function get_model_rotations_r_II(x)
        d(vcat(x...))
    end
    append!(params_objects, [d])
    return get_model_rotations_r_II
end

eco.comps["r_II_1"].model = identity
for i=2:19
    eco.comps[string("r_II_",i)].model = get_model_rotations_r_II(
        length(filter(x -> x == i, clusters)) + 
        length(filter(x -> x == i-1, clusters)),
        length(filter(x -> x == i, clusters)) + 
        length(filter(x -> x == i-1, clusters))
    )
end


########################################################
# Hidden dense

function get_model_dense(in::Int64, out::Int64)
    d1 = Dense(in*2, 100)
    d2 = Dense(100, 10)
    d3 = Dense(10, out)
    function get_model_dense(x)
        x = Flux.flatten(reshape(x, (size(x)..., 1)))
        a = real.(x)
        b = imag.(x)
        x = vcat(a, b)
        d3(d2(d1(x)))
    end
    append!(params_objects, [d1, d2, d3])
    return get_model_dense
end

for i=1:19
    eco.comps[string("h_",i)].model = get_model_dense(
        length(filter(x -> x == i, clusters)) + 
        length(filter(x -> x == i-1, clusters)),
        size(labels)[1]
    )
end

########################################################
# Output

function get_model_output(in::Int64, out::Int64)
    d = Dense(in, out)
    function get_model_output(x)
        d(vcat(x...))
    end
    append!(params_objects, [d])
    return get_model_output
end

eco.comps["out1"].model = get_model_output(
    length(unique(clusters)) * size(labels)[1],
    size(labels)[1]
)


########################################################
# package Maen model

eco.schc = scv(eco.comps, eco.sch)
eco.ps_obj = params_objects

function nn(input_data::Any)
    reduce(vcat, Maen.model(eco, input_data)[end])
end
