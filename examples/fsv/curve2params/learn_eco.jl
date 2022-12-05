
using Maen
using Flux, StatsBase, LinearAlgebra
using ThreadTools

batchrun = s -> reduce(hcat, map(x -> nn(x), s))
loss = (x, y) -> Flux.mse(batchrun(x), y)
acc = (x, y) -> median(((abs.(batchrun(x).-y))./y))
accp = (x, y) -> median(((abs.(batchrun(x).-y))./y), dims=2)

cb = () -> println("acc = (", acc(minibatch()...), ", ", acc(testbatch()...), " )",
    " loss = (", loss(minibatch()...), ", ", loss(testbatch()...), " )")

Flux.Optimise.train!(
    loss, Flux.params(eco.ps_obj), 
    repeatedly(minibatch, 500), ADAM(),
    cb = Flux.throttle(cb, 10)
)

println("total train: accuracy = ", acc(minibatch()...), " loss = ", loss(minibatch()...));
println("total test: accuracy = ", acc(testbatch()...), " loss = ", loss(testbatch()...));

println(sum(length, Flux.params(eco.ps_obj)))
