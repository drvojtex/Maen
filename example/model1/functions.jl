
using Flux

params_objcts = []

function get_model_dense()
    d = Dense(1, 1)
    function get_model_dense(x)
        return d(x)
    end
    append!(params_objcts, [d])
    return get_model_dense
end
model_f = get_model_dense()
model_g = get_model_dense()
model_h = get_model_dense()
model_q = get_model_dense()

model_p(x, y) = x .+ y
model_n(x, y) = x .+ y

model_params = Flux.params(params_objcts)

model(x) = model_n(model_p(model_f(identity(x[1])), model_g(identity(x[1]))), model_q(model_h(identity(x[2]))))