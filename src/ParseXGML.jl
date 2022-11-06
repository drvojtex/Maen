
using LibExpat

function xgml2network(path::String)
    net::Network = Network()
    
    f::IOStream = open(path, "r")
    xgml::ETree = LibExpat.xp_parse(read(f, String))
    close(f)

    for node in collect(xgml[xpath"/section/section/section[@name='node']"])
        node::ETree = xp_parse(string(node))
        label::Vector{String} = split(node[
            xpath"/section/attribute[@key='label']/text()"
        ][1], ' ')  
        append!(net.ids, parse(Int, node[
            xpath"/section/attribute[@key='id']/text()"
        ][1])+1)
        append!(net.types, [label[1]])
        append!(net.names, [label[2]])
    end

    dict_append! = (d, key, value) -> 
        haskey(d, key) ? append!(d[key], [value]) : d[key]=[value]

    for edge in collect(xgml[xpath"/section/section/section[@name='edge']"])
        source::Int64 = parse(Int, xp_parse(string(edge))[
            xpath"/section/attribute[@key='source']/text()"
        ][1])
        target::Int64 = parse(Int, xp_parse(string(edge))[
            xpath"/section/attribute[@key='target']/text()"
        ][1])
        dict_append!(net.adjacency_list, source+1, target+1)
    end
    
    return net
end
