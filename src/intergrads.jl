
using Flux, Zygote
using AbstractDifferentiation, NumericalIntegration
using ProgressBars


∂m(m::Function, x::Any) = map(d -> abs.(d), AD.gradient(AD.ZygoteBackend(), m, x)[1])
∂x(m::Function, x::Any) = mapreduce(样本->∂m(m, 样本), .+, eachcol(x)) 

function intergrads(data::Any, labels::Any, m::Function; step=0.005)
    @info "Prepare data"
    d = []
    Threads.@threads for i=1:length(labels)
        if (m(data[:,i]) > .5) == labels[i]
            append!(d, [data[:,i]])
        end
    end
    d = permutedims(mapreduce(permutedims, vcat, d))
    x_data = [d*λ for λ::Float32 in ProgressBar(0:step:1)]
    
    @info "Compute gradients"
    y_data = [] 
    Threads.@threads for i in ProgressBar(1:Int(1/step)+1)
        append!(y_data, [∂x(x -> m(x), x_data[i])])
    end
    @info "Integrate"
    integrate(0:step:1, y_data)
end
