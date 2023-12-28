from abc import abstractmethod
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.keras.initializers import Initializer

from typing import Optional

INIT_TECHNIQUES = {'zero_imag', 'mirror'}


def _calculate_fan_dimensions(shape):
    """
    Computes the number of input and output units for a weight shape.
    Taken from TensorFlow's implementation (link in the original comment).

    Args:
        shape: Integer shape tuple or TF tensor shape.

    Returns:
        A tuple of scalars (fan_in, fan_out).
    """
    fan_in, fan_out = 1, 1  # Default values for constants or empty shapes.

    if len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in, fan_out = shape[0], shape[1]
    elif len(shape) > 2:
        # For convolutional layers (2D, 3D, etc.)
        # kernel shape: (..., input_depth, depth)
        receptive_field_size = np.prod(shape[:-2])
        fan_in, fan_out = shape[-2] * receptive_field_size, shape[-1] * receptive_field_size

    return fan_in, fan_out


class _StochasticDistributionsFactory:
    """
    Random generator that selects appropriate random ops.
    Reference: TensorFlow implementation (URL in the original comment).
    """

    def __init__(self, seed=None):
        self.seed = [seed, 0] if seed is not None else None

    def _generate_random_op(self, op_stateless, op_stateful, shape, **kwargs):
        """Helper function to select and use the appropriate random operation."""
        op = op_stateless if self.seed else op_stateful
        return op(shape=shape, seed=self.seed, **kwargs)

    def random_normal(self, shape, mean=0.0, stddev=1, dtype=tf.dtypes.float32):
        """Generates a deterministic random normal distribution if seed is provided."""
        return self._generate_random_op(stateless_random_ops.stateless_random_normal,
                                        random_ops.random_normal,
                                        shape, mean=mean, stddev=stddev, dtype=dtype)

    def random_uniform(self, shape, minval, maxval, dtype):
        """Generates a deterministic random uniform distribution if seed is provided."""
        return self._generate_random_op(stateless_random_ops.stateless_random_uniform,
                                        random_ops.random_uniform,
                                        shape, minval=minval, maxval=maxval, dtype=dtype)

    def truncated_normal(self, shape, mean, stddev, dtype):
        """Generates a deterministic truncated normal distribution if seed is provided."""
        return self._generate_random_op(stateless_random_ops.stateless_truncated_normal,
                                        random_ops.truncated_normal,
                                        shape, mean=mean, stddev=stddev, dtype=dtype)



class ComplexWeightInitializer(Initializer):
    """
    Initializer for complex-valued neural network weights.

    Supports uniform and normal distributions for the initialization.

    Args:
        distribution (str): Type of distribution to use ('uniform' or 'normal').
        seed (Optional[int]): Seed for random number generation.
    """
    def __init__(self, distribution: str = "uniform", seed: Optional[int] = None):
        if distribution.lower() not in {"uniform", "normal"}:
            raise ValueError("Invalid `distribution` argument:", distribution)
        else:
            self.distribution = distribution.lower()
        self._random_generator = _StochasticDistributionsFactory(seed)

    def _call_random_generator(self, shape, arg, dtype):
        if self.distribution == "uniform":
            return self._random_generator.random_uniform(shape=shape, minval=-arg, maxval=arg, dtype=dtype)
        elif self.distribution == "normal":
            # I make this magic number division because that's what tf does on this case
            return self._random_generator.truncated_normal(shape=shape, mean=0.0, stddev=arg / .87962566103423978,
                                                           dtype=dtype)

    @abstractmethod
    def _compute_limit(self, fan_in, fan_out):
        pass

    def __call__(self, shape, dtype=tf.dtypes.complex64, **kwargs):
        fan_in, fan_out = _calculate_fan_dimensions(shape)
        arg = self._compute_limit(fan_in, fan_out)
        dtype = tf.dtypes.as_dtype(dtype)
        if dtype.is_complex:
            arg = arg / np.sqrt(2)
        return self._call_random_generator(shape=shape, arg=arg, dtype=dtype.real_dtype)

    def get_config(self):  # To support serialization
        return {"seed": self._random_generator.seed}


class ComplexXavierInitializer(ComplexWeightInitializer):
    """
    Xavier (Glorot) initializer for complex-valued weights.

    It initializes the weights with values drawn from a uniform distribution 
    scaled according to the Xavier (Glorot) method, which is effective for 
    maintaining activation variances in deep networks.

    Usage:
    - Standalone: `initializer = ComplexXavierInitializer(); values = initializer(shape=(2, 2))`
    - In a cvnn layer: `layer = layers.ComplexDense(units=10, kernel_initializer=ComplexXavierInitializer())`

    Args:
        seed (Optional[int]): Seed for random number generation.
    """

    def __init__(self, seed: Optional[int] = None):
        super(ComplexXavierInitializer, self).__init__(distribution="uniform", seed=seed)

    def _compute_limit(self, fan_in, fan_out):
        return tf.math.sqrt(6. / (fan_in + fan_out))
    
class Zeros:
    """
    Creates a tensor with all elements set to zero.

    ```
    > >> initializers.Zeros()(shape=(2,2))
    <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
    array([[0.+0.j, 0.+0.j],
          [0.+0.j, 0.+0.j]], dtype=float32)>
    ```

    ```
    # Usage in a cvnn layer:
    layer = layers.ComplexDense(units=10, bias_initializer=initializer)
    ```
    """
    def __call__(self, shape, dtype=tf.dtypes.complex64):
        return tf.zeros(shape, dtype=tf.dtypes.as_dtype(dtype).real_dtype)


class Ones:
    def __call__(self, shape, dtype=tf.dtypes.complex64):
        return tf.ones(shape, dtype=tf.dtypes.as_dtype(dtype).real_dtype)