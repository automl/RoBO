import theano.tensor as T
import lasagne

from robo.models.lcnet import basis_functions


class BasisFunctionLayer(lasagne.layers.MergeLayer):
    def __init__(self, incoming, **kwargs):
        super(BasisFunctionLayer, self).__init__(incoming, **kwargs)

    def get_output_for(self, inputs, **kwargs):
        t = inputs[0][:, 0]

        a = inputs[1][:, 0]
        b = inputs[1][:, 1]
        c = inputs[1][:, 2]
        y_a = basis_functions.vapor_pressure(t, a, b, c)

        a = inputs[1][:, 3]
        b = inputs[1][:, 4]
        y_b = basis_functions.pow_func(t, a, b)

        a = inputs[1][:, 5]
        b = inputs[1][:, 6]
        c = inputs[1][:, 7]
        y_c = basis_functions.log_power(t, a, b, c)

        a = inputs[1][:, 8]
        b = inputs[1][:, 9]
        y_d = basis_functions.exponential(t, a, b)

        a = inputs[1][:, 10]
        b = inputs[1][:, 11]
        c = inputs[1][:, 12]
        y_e = basis_functions.hill_3(t, a, b, c)

        l = [y_a, y_b, y_c, y_d, y_e]

        y = T.stack(*l)

        return y.T

    def get_output_shape_for(self, input_shape):
        return input_shape[0][0], 5
