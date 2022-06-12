
using Plots
using Graphs, GraphRecipes

struct Component
    id::Int64
    model::Function
    input_id::Int64
end

el = Edge.([(6, 3), (3, 1), (1, 4), (1, 2), (7, 5), (5, 2), (2, 4)])
graph = SimpleDiGraph(el)
tmp = deepcopy(graph.badjlist)
map(x->append!(x, -1), tmp)

function get_complex_function(top::Int64, badjlist::Vector{Vector{Int64}}, 
            components::AbstractArray{Component})
    s = "model(x) = "
    b = false
    function dfs_rec(v)
        if v == -1
            b = true
            s *= ")"
            return
        end
        if b s *= ", " end
        b = false
        s *= "$(string(nameof(filter(x->x.id == v, components)[1].model)))("
        if badjlist[v][1] == -1 s *= "x[$(components[v].input_id)]" end
        for w in badjlist[v] dfs_rec(w) end
    end
    dfs_rec(top)
    println(s)
    return eval(Meta.parse(s))
end

f(x) = 2*x
g(x, y) = x+y
k(x) = x + 1
h(x, y) = x+2*y
l(x) = x+2
n(x) = x
p(x) = x

comp = [
    Component(1, f, -1),
    Component(2, g, -1),
    Component(3, k, -1),
    Component(4, h, -1),
    Component(5, l, -1),
    Component(6, n, 1),
    Component(7, p, 2)
]

m = get_complex_function(4, tmp, comp)

m([1, 1])

graphplot(graph, names=map(x->string(nameof(x.model)), comp))
