
using DocumentFunction

@doc """
$(DocumentFunction.documentfunction(xgml2network;
    location=false,
    maintext="Parse yEd xgml file into the Network struct.",
    argtext=Dict("path"=>"path to the xgml file")))
""" xgml2network
