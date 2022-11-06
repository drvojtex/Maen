
using DocumentFunction

@doc """
$(DocumentFunction.documentfunction(create_ecosystem;
    location=false,
    maintext="Create the Ecosystem struct from the xgml setup file.",
    argtext=Dict("sf"=>"path to the configuration XGML created in yEd editor")))
""" create_ecosystem

@doc """
$(DocumentFunction.documentfunction(create_graph;
    location=false,
    maintext="Create graph in Graphs representation from the Network struct.",
    argtext=Dict("net"=>"Network struct")))
""" create_graph

@doc """
$(DocumentFunction.documentfunction(create_components;
    location=false,
    maintext="Create components structures.",
    argtext=Dict("net"=>"Network struct",
                 "graph"=>"graph of components connections")))
""" create_components

@doc """
$(DocumentFunction.documentfunction(scheduling;
    location=false,
    maintext="Get the scheduling vector.",
    argtext=Dict("g"=>"graph of the network connections")))
""" scheduling

@doc """
$(DocumentFunction.documentfunction(scv;
    location=false,
    maintext="Schedule components by a scheduling vector.",
    argtext=Dict("components"=>"components of the network",
                 "sch"=>"scheduling vector")))
""" scv

@doc """
$(DocumentFunction.documentfunction(model_output;
    location=false,
    maintext="Get output of the particular component.",
    argtext=Dict("eco"=>"ecosystem of the network",
                 "c"=>"current component",
                 "data"=>"input network data",
                 "values"=>"previous scheduled components outputs")))
""" model_output

@doc """
$(DocumentFunction.documentfunction(model;
    location=false,
    maintext="Get outputs of all components.",
    argtext=Dict("eco"=>"ecosystem of the network",
                 "data"=>"input network data")))
""" model

@doc """
$(DocumentFunction.documentfunction(subset_model;
    location=false,
    maintext="Compute outputs of components with usage o fsubset of components.",
    argtext=Dict("eco"=>"ecosystem of the network",
                 "data"=>"input network data",
                 "subset"=>"subset of components ids to be taken in computation"),
    keytext=Dict("noise"=>"if true, nonused components streams are replaced by a noise, else (default) are replaced by zeros")))
""" subset_model
