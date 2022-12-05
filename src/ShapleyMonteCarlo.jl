
using Combinatorics, LinearAlgebra
using Distributions
using ProgressBars


function get_U(d::Int64)
    U::Matrix{Int64} = zeros(d-1, d)
    for i=1:d-1 for j=1:i
        U[i, j] = 1
    end end
    for i=1:d-1
        U[i, i+1] = -i
    end    
    return U
end

function get_perms_members(d::Int64)
    v = Vector{Float64}(1:d)
    v .-= (d+1)/2
    v ./= sqrt(norm(Vector{Int64}(1:d)))
    return v
end

function argsort(x::Vector{Float64}, p::Vector{Float64})
    d::Int64 = length(x)
    x_tmp::Vector{Float64} = sort(x)
    tmp_result::Vector{Float64} = zeros(d)
    for i=1:d
        tmp_result[findall(y -> y == x_tmp[i], x)[1]] = p[i]
    end
    tmp_result = Int.(round.(tmp_result .* sqrt(norm(Vector{Int64}(1:d))) .+ (d+1)/2))
    result::Vector{Int64} = zeros(d)
    for i=1:d
        result[i] = findall(y -> y == i, tmp_result)[1]
    end
    return result
end

function get_random_permutation(U::Matrix{Int64}, p::Vector{Float64})
    d::Int64 = length(p)
    x::Vector{Float64} = rand(Uniform(-1, 1), d-1)
    x ./= norm(x)
    argsort(U'*x, p)
end

function get_random_subset(U::Matrix{Int64}, p::Vector{Float64}, i::Int64)
    perm::Vector{Int64} = get_random_permutation(U, p)
    perm[1:findall(y -> y==i, perm)[1]-1]
end

function generate_subsets(d::Int64, k::Int64, i::Int64)
    U::Matrix{Int64} = get_U(d)
    p::Vector{Float64} = get_perms_members(d)
    result = []
    for _ in ProgressBar(1:k)
        tmp::Vector{Int64} = get_random_subset(U, p, i)
        if length(tmp) > 0
            append!(result, [Set(tmp)])
        end
    end
    return unique(result)
end
