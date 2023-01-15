
using Flux, Zygote
using AbstractDifferentiation, NumericalIntegration


∂m(m::Function, x::Any) = map(d -> broadcast.(abs, d), AD.gradient(AD.ZygoteBackend(), m, x)[1])

"""
    mapreduce(i->intergrads(data[i], x->loss(x, labels[i])), .+, 1:100)
"""
function intergrads(data::Any, f::Function; step=0.005)
    #@info "Prepare data"
    x_data = [λ .* data for λ::Float32 in 0:step:1]
    #@info "Compute gradients"
    g_data = ∂m(f, x_data)
    #@info "Integrate"
    integrate(0:step:1, g_data)
end
