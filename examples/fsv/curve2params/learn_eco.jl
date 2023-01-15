
using Maen
using Flux, StatsBase, LinearAlgebra
using ThreadTools

batchrun = s -> reduce(hcat, map(x -> nn(x), s))
loss = (x, y) -> Flux.mse(batchrun(x), y)
acc = (x, y) -> median(((abs.(batchrun(x).-y))./y))
accp = (x, y) -> median(((abs.(batchrun(x).-y))./y), dims=2)

acc_eco = []
loss_eco = []

cb = () -> (
    println("acc = (", acc(minibatch()...), ", ", acc(testbatch()...), " )",
            " loss = (", loss(minibatch()...), ", ", loss(testbatch()...), " )"),
    append!(acc_eco,[(acc(minibatch()...), acc(testbatch()...))]),
    append!(loss_eco,[(loss(minibatch()...), loss(testbatch()...))])
)

Flux.Optimise.train!(
    loss, Flux.params(eco.ps_obj), 
    repeatedly(minibatch, 1000), ADAM(),
    cb = Flux.throttle(cb, 0.1)
)

println("total train: accuracy = ", acc(minibatch()...), " loss = ", loss(minibatch()...));
println("total test: accuracy = ", acc(testbatch()...), " loss = ", loss(testbatch()...));

println(sum(length, Flux.params(eco.ps_obj)))
