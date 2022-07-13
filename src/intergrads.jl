
using Flux
using Zygote, AbstractDifferentiation

∂m(m::Function, x::Any) = AD.gradient(AD.ZygoteBackend(), m, x)[1]
∂x(m::Function, x::Any) = mapreduce(样本->∂m(m, 样本), .+, eachcol(x)) 

#=
using NumericalIntegration
x_data = [data*Float32(l) for l=0:0.01:1];
y_data = map(d->∂x(x->model(eco,x)[end][1],d),x_data);
integrate(x_data,y_data)
=#
