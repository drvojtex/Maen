
using Maen
using Graphs, SimpleWeightedGraphs
using Flux, StatsBase, LinearAlgebra


# Load xgml file and setup inputs mapping
eco = create_ecosystem("data/network.xgml");
eco.ii = Dict( 
    map(x -> x.id, sort(filter(x->typeof(x)==Component{InputAgent}, collect(values(eco.comps))), by=x->parse(Int, replace(x.name, "in"=>""))))
    .=>
    1:12
);

# Vector of trainable objects
params_objects = []


########################################################
# Rotations

function get_model_rotations(dims::Int64)
    
    d1 = Dense(dims, dims)

    function get_model_rotations(x)
        d1(x)
    end
    append!(params_objects, [d1])
    return get_model_rotations
end
for i=1:12
    eco.comps[string("r_",i)].model = 
        get_model_rotations(length(filter(x -> x == i, clusters)))
end


########################################################
# Laurent polynom
#=
function get_model_laurent(smoothness::Int64)
    a::Matrix{ComplexF64} = rand(Complex{Float64}, 1, smoothness*2+1)
    c::ComplexF64 = rand(Complex{Float64})
    r::StepRange{Int64, Int64} = -smoothness:1:smoothness
    laurent_p = z -> reduce(vcat, 
        Flux.tanh.(a * map(n -> (Complex{Float64}(z)-c)^-n, r))
    )
    function get_model_laurent(x)
        laurent_p.(x)
    end
    append!(params_objects, [a, c])
    return get_model_laurent
end
=#

function get_model_laurent()
    function get_model_laurent(x)
        vcat(x...)
    end
    return get_model_laurent
end
eco.comps["lp_1"].model = identity
for i=2:12
    eco.comps[string("lp_",i)].model = get_model_laurent()
end


########################################################
# Hidden dense

function get_model_dense(in::Int64, out::Int64)
    d1 = Dense(in*2, 100)
    d2 = Dense(100, 50)
    d3 = Dense(50, 10)
    d4 = Dense(10, out)
    function get_model_dense(x)
        x = Flux.flatten(reshape(x, (size(x)..., 1)))
        d4(d3(d2(d1(x))))
    end
    append!(params_objects, [d1, d2, d3, d4])
    return get_model_dense
end

for i=1:12
    eco.comps[string("h_",i)].model = get_model_dense(
        length(filter(x -> x == i, clusters)) + 
        length(filter(x -> x == i-1, clusters)),
        size(labels)[1]
    )
end

########################################################
# Output

function get_model_output(in::Int64, out::Int64)
    d1 = Dense(in, out)
    function get_model_output(x)
        d1(vcat(x...))
    end
    append!(params_objects, [d1])
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
