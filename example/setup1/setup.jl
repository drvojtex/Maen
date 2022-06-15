
using Flux

params_objects = []

function get_model_lstm()
    l = LSTM(100, 10)
    function get_model_lstm(x)
        return l(x)
    end
    append!(params_objects, [l])
    return get_model_lstm
end
model_o = get_model_lstm()
model_c = get_model_lstm()
model_h = get_model_lstm()
model_l = get_model_lstm()

function get_model_dense()
    d = Dense(40, 1)
    function get_model_dense(x1, x2, x3, x4)
        x = vcat(x1, x2, x3, x4)
        return d(x)
    end
    append!(params_objects, [d])
    return get_model_dense
end
model_n = get_model_dense()

model_params = Flux.params(params_objects)
