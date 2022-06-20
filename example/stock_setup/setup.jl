
using Flux

params_objects = []

function get_model_lstm()
    l = LSTM(50, 1)
    function get_model_lstm(x)
        return l(x)
    end
    append!(params_objects, [l])
    return get_model_lstm
end
lstm_o = get_model_lstm()
lstm_c = get_model_lstm()
lstm_h = get_model_lstm()
lstm_l = get_model_lstm()

function get_model_dense(in_dim::Int64)
    d = Dense(in_dim, 1)
    function get_model_dense(x...)
        x = vcat(x...)
        return d(x)
    end
    append!(params_objects, [d])
    return get_model_dense
end
model_n = get_model_dense(9)
dense_h = get_model_dense(4*50)

function get_stats_model()
    function get_stats_model(x)
        return vcat(mean(x, dims=1), var(x, dims=1), ([1:50 ones(50)]\x))
    end
    return get_stats_model
end
stats = get_stats_model()

model_params = Flux.params(params_objects)
