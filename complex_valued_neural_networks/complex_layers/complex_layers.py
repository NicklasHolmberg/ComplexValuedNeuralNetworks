from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import initializers
from tensorflow import TensorShape, Tensor

from typing import Union, List
from complex_valued_neural_networks.activation_functions import t_activation
from complex_valued_neural_networks.complex_initializers import ComplexXavierInitializer, Zeros, ComplexWeightInitializer, INIT_TECHNIQUES


t_input = Union[Tensor, tuple, list]
t_input_shape = Union[TensorShape, List[TensorShape]]

DEFAULT_COMPLEX_TYPE = tf.as_dtype(np.complex64)


class ComplexLayer(ABC):

    @abstractmethod
    def get_real_equivalent(self):
        """
        :return: Gets a real-valued COPY of the Complex Layer.
        """
        pass

class ComplexFlatten(Flatten, ComplexLayer):

    def call(self, inputs: t_input):
        real_flat = super(ComplexFlatten, self).call(tf.math.real(inputs))
        imag_flat = super(ComplexFlatten, self).call(tf.math.imag(inputs))
        return tf.cast(tf.complex(real_flat, imag_flat), inputs.dtype)  # Keep input dtype

    def get_real_equivalent(self):
        return ComplexFlatten(name=self.name + "_real_equiv")


class ComplexDense(Dense, ComplexLayer):
    """
    A fully connected layer for complex-valued inputs.

    This layer applies an element-wise activation function to its inputs 
    after performing a weighted sum with an optional bias.
    """

    def __init__(self, units: int, activation: t_activation = None, use_bias: bool = True,
                 kernel_initializer=ComplexXavierInitializer,
                 bias_initializer=Zeros,
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 dtype=DEFAULT_COMPLEX_TYPE,
                 init_technique: str = 'mirror',
                 **kwargs):
        """
        Initializes the ComplexDense layer.

        Args:
            units (int): Number of output units.
            activation (t_activation): Activation function for the layer. Defaults to linear if not specified.
            use_bias (bool): Whether to use a bias vector.
            kernel_initializer: Initializer for the weights matrix. Defaults to ComplexXavierInitializer.
            bias_initializer: Initializer for the bias vector. Defaults to Zeros.
            dtype: Data type of the layer. Defaults to DEFAULT_COMPLEX_TYPE.
            init_technique (str): Technique to initialize complex numbers ('mirror' or 'zero_imag').
                'mirror': Same initializer for real and imaginary parts.
                'zero_imag': Initialize real part only, imaginary part set to zero.
        """
        if activation is None:
            activation = "linear"
        super(ComplexDense, self).__init__(units, activation=activation, use_bias=use_bias,
                                           kernel_initializer=kernel_initializer,
                                           bias_initializer=bias_initializer,
                                           kernel_constraint=kernel_constraint, kernel_regularizer=kernel_regularizer,
                                           **kwargs)
        self.my_dtype = tf.dtypes.as_dtype(dtype)
        self.init_technique = init_technique.lower()

    def build(self, input_shape):
        if self.my_dtype.is_complex:
            i_kernel_dtype = self.my_dtype if isinstance(self.kernel_initializer,
                                                         ComplexWeightInitializer) else self.my_dtype.real_dtype
            i_bias_dtype = self.my_dtype if isinstance(self.bias_initializer,
                                                       ComplexWeightInitializer) else self.my_dtype.real_dtype
            i_kernel_initializer = self.kernel_initializer
            i_bias_initializer = self.bias_initializer
            if not isinstance(self.kernel_initializer, ComplexWeightInitializer):
                tf.print(f"WARNING: you are using a Tensorflow Initializer for complex numbers. "
                         f"Using {self.init_technique} method.")
                if self.init_technique in INIT_TECHNIQUES:
                    if self.init_technique == 'zero_imag':
                        i_kernel_initializer = initializers.Zeros()
                        i_bias_initializer = initializers.Zeros()
                else:
                    raise ValueError(f"Unsuported init_technique {self.init_technique}, "
                                     f"supported techniques are {INIT_TECHNIQUES}")

            self.w_r = self.add_weight('kernel_r',
                                     shape=(input_shape[-1], self.units),
                                     dtype=self.my_dtype.real_dtype,
                                     initializer=self.kernel_initializer,
                                     trainable=True,
                                     constraint=self.kernel_constraint, regularizer=self.kernel_regularizer)
            self.w_i = self.add_weight('kernel_i',
                                     shape=(input_shape[-1], self.units),
                                     dtype=self.my_dtype.real_dtype,
                                     initializer=self.kernel_initializer,
                                     trainable=True,
                                     constraint=self.kernel_constraint, regularizer=self.kernel_regularizer)
            if self.use_bias:
                self.b_r = tf.Variable(
                    name='bias_r',
                    initial_value=self.bias_initializer(shape=(self.units,), dtype=i_bias_dtype),
                    trainable=self.use_bias
                )
                self.b_i = tf.Variable(
                    name='bias_i',
                    initial_value=i_bias_initializer(shape=(self.units,), dtype=i_bias_dtype),
                    trainable=self.use_bias
                )
        else:
            self.w = self.add_weight('kernel',
                                     shape=(input_shape[-1], self.units),
                                     dtype=self.my_dtype,
                                     initializer=self.kernel_initializer,
                                     trainable=True,
                                     constraint=self.kernel_constraint, regularizer=self.kernel_regularizer)
            if self.use_bias:
                self.b = self.add_weight('bias', shape=(self.units,), dtype=self.my_dtype,
                                         initializer=self.bias_initializer, trainable=self.use_bias)

    def call(self, inputs: t_input):
        if inputs.dtype != self.my_dtype:
            tf.print(f"WARNING: {self.name} - Expected input to be {self.my_dtype}, but received {inputs.dtype}.")
            if self.my_dtype.is_complex and inputs.dtype.is_floating:
                tf.print("\tThis is normally fixed using ComplexInput() "
                         "at the start (tf casts input automatically to real).")
            inputs = tf.cast(inputs, self.my_dtype)
        if self.my_dtype.is_complex:
            w = tf.complex(self.w_r, self.w_i)
            if self.use_bias:
                b = tf.complex(self.b_r, self.b_i)
        else:
            w = self.w
            if self.use_bias:
                b = self.b
        out = tf.matmul(inputs, w)
        if self.use_bias:
            out = out + b
        return self.activation(out)

    def get_real_equivalent(self, output_multiplier=2):
        return ComplexDense(units=int(round(self.units * output_multiplier)),
                            activation=self.activation, use_bias=self.use_bias,
                            kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer,
                            kernel_constraint=self.kernel_constraint, kernel_regularizer=self.kernel_regularizer,
                            dtype=self.my_dtype.real_dtype, name=self.name + "_real_equiv")

    def get_config(self):
        config = super(ComplexDense, self).get_config()
        config.update({
            'dtype': self.my_dtype,
            'init_technique': self.init_technique

        })
        return config