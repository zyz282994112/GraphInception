from keras.layers import Layer, InputSpec, merge
from keras import regularizers, initializations, activations, constraints
from keras import backend as K
import numpy as np

class GraphHighway(Layer):
    def __init__(self, init='glorot_uniform', transform_bias=-2,
                 n_rel=1,activation='linear', weights=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, input_dim=None, **kwargs):
        self.init = initializations.get(init)
        self.transform_bias = transform_bias
        self.activation = activations.get(activation)
        self.n_rel = n_rel

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(GraphHighway, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = self.input_dim
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        self.W = self.init((input_dim, input_dim),
                           name='{}_W'.format(self.name))
        self.W_carry = self.init((input_dim, input_dim),
                                 name='{}_W_carry'.format(self.name))
        self.V = self.init((self.n_rel*input_dim, input_dim),
                           name='{}_V'.format(self.name))

        if self.bias:
            self.b = K.zeros((input_dim,), name='{}_b'.format(self.name))
            # initialize with a vector of values `transform_bias`
            self.b_carry = K.variable(np.ones((input_dim,)) * self.transform_bias,
                                      name='{}_b_carry'.format(self.name))
            self.V_carry = self.init((self.n_rel*input_dim, input_dim),
                             name='{}_V_carry'.format(self.name))

            self.trainable_weights = [self.W, self.V, self.b, self.W_carry, self.V_carry, self.b_carry]

        else:
            self.V_carry = self.init((self.n_rel*input_dim, input_dim),
                             name='{}_V_carry'.format(self.name))

            self.trainable_weights = [self.W, self.W_carry, self.V, self.V_carry]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.bias and self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint
        if self.bias and self.b_constraint:
            self.constraints[self.b] = self.b_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def get_output_shape_for(self, input_shape):
        return (None, self.input_dim)

    def call(self, inputs, mask=None):
        x = inputs[0] #feature matrix
        rel = inputs[1] # n_nodes, n_rel, n_neigh

        # dot(V_carry, context)
        carry_gate = K.dot(x, self.W_carry)
        carry_context = K.dot(rel,self.V_carry)
        carry_gate += carry_context

        if self.bias:
             carry_gate += self.b_carry
        carry_gate = activations.sigmoid(carry_gate)

        # dot(V, context)
        context = K.dot(rel,self.V)
        h = K.dot(x, self.W) + context
        if self.bias:
            h += self.b

        h = self.activation(h)
        h = carry_gate * h + (1 - carry_gate) * x
        return h

    def get_config(self):
        config = {'init': self.init.__name__,
                  'transform_bias': self.transform_bias,
                  'activation': self.activation.__name__,
                  'n_rel': self.n_rel,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias,
                  'input_dim': self.input_dim}
        base_config = super(GraphHighway, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
