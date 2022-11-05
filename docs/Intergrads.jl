
using DocumentFunction

@doc """
$(DocumentFunction.documentfunction(intergrads;
    location=false,
    maintext="Compute importnce of the input data parts by the intergrads method.",
    argtext=Dict("data"=>"input data of the network",
                 "labels"=>"labels corresponding to the data",
                 "m"=>"network model function"),
    keytext=Dict("step"=>"step of the gradient")))
""" intergrads
