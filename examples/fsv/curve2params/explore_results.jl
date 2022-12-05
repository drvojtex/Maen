
function rerr(i::Int64)
    tmp::Vector{Float64} = (abs.((batchrun(data).-labels))./labels)[i,:]
    tmp[findall(x -> x > 1, tmp)] .= 1
    return tmp
end

#=
p = [
    scatter(1:100, rerr(1), label="Ec"),
    scatter(1:100, rerr(2), label="Fc"),
    scatter(1:100, rerr(3), label="Ft"),
    scatter(1:100, rerr(4), label="Gf")
]
=#

#plot(1:100, rerr(1), label="Ec")
#plot!(p, 1:100, rerr(2), label="Fc")
p=plot(1:100, rerr(3), label="Ft")
plot!(p, 1:100, rerr(4), label="Gf")

plot(p)

