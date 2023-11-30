import tensorflow as tf
from typing import Union, Callable, Optional
from tensorflow import Tensor

t_activation = Union[str, Callable]

"""
Complex Valued activation function
"""

def complex_tanh(z: tf.Tensor) -> tf.Tensor:
    """
    Complex Hyperbolic Tangent (Tanh) Activation Function
    ---------------------------------------------------
    - Description: Applies the hyperbolic tangent (tanh) activation function to both the real and imaginary parts of a complex tensor.
    - Parameters: 
      - z: A TensorFlow tensor of complex numbers.
    - Functionality: 
      - Separately computes the tanh for the real and imaginary components of 'z'.
      - Constructs a new complex tensor from the tanh-transformed real and imaginary parts.
    - Usage: Suitable for introducing non-linearity in neural networks processing complex-valued data.
    """
    real_tanh = tf.math.tanh(tf.math.real(z))
    imag_tanh = tf.math.tanh(tf.math.imag(z))
    return tf.complex(real_tanh, imag_tanh)

def complex_cardioid(z: tf.Tensor) -> tf.Tensor:
    """
    complex_cardioid Function (TYPE A: Cartesian Form)
    --------------------------------------------------
    - Source: "Better than Real: Complex-valued Neural Nets for MRI Fingerprinting" by V. Patrick, 2017.
    - Description: The complex_cardioid function attenuates the magnitude of a complex input 'z' based on its phase, while preserving the phase information.
    - Behavior:
      - For complex inputs, it modifies the magnitude while keeping the phase unchanged.
      - For real-valued inputs, the behavior is analogous to the ReLU function.
    - Parameters:
      - z: Input complex tensor.
    - Returns: A complex tensor with the cardioid activation applied, adjusting magnitude based on phase.
    """
    cardioid_factor = 1 + tf.math.cos(tf.math.angle(z))
    return tf.cast(cardioid_factor, dtype=z.dtype) * z / 2


def complex_leaky_relu(z: tf.Tensor, alpha: float = 0.3):
    """
    Complex Leaky ReLU Activation Function
    --------------------------------------
    - Description: Applies the Leaky ReLU activation function to both the real and imaginary parts of a complex tensor.
    - Parameters: 
    - z: A TensorFlow tensor of complex numbers.
    - alpha: The negative slope coefficient for the Leaky ReLU function (default is 0.3).
    - Functionality: 
    - Separately computes the Leaky ReLU for the real and imaginary components of 'z'.
    - Constructs a new complex tensor from the Leaky ReLU-transformed real and imaginary parts.
    - Usage: Suitable for introducing non-linearity in neural networks processing complex-valued data.
    """
    real = tf.math.real(z)
    imag = tf.math.imag(z)
    return tf.complex(tf.keras.layers.LeakyReLU(alpha)(real), tf.keras.layers.LeakyReLU(alpha)(imag))


def zrelu(z: Tensor, epsilon=1e-7) -> Tensor:
    """
    zReLU Activation Function
    -------------------------
    - Source: "On Complex Valued Convolutional Neural Networks" by Nitzan Guberman, 2016.
    - Description: zReLU maintains the input as the output when both real and imaginary 
    parts of the input are positive.
    - Application: Useful in complex-valued neural networks.
    - Further Reading on Custom Activation Functions in Keras/TensorFlow:
    Stack Overflow: https://stackoverflow.com/questions/49412717/advanced-custom-activation-function-in-keras-tensorflow
    """
    imag_relu = tf.nn.relu(tf.math.imag(z))
    real_relu = tf.nn.relu(tf.math.real(z))
    ret_real = imag_relu*real_relu / (imag_relu + epsilon)
    ret_imag = imag_relu*real_relu / (real_relu + epsilon)
    ret_val = tf.complex(ret_real, ret_imag)
    return ret_val


def modrelu(z: tf.Tensor, b: float = 1., c: float = 1e-3) -> tf.Tensor:
    """
    mod ReLU Activation Function
    ----------------------------
    - Source: "Unitary Evolution Recurrent Neural Networks" by M. Arjovsky et al., 2016.
    - URL: https://arxiv.org/abs/1511.06464
    - Description: modReLU, a variant of the ReLU, is a pointwise nonlinearity that operates on complex numbers.
    - Functionality: modReLU(z) = ReLU(|z| + b) * z / |z|
    It modifies only the absolute value of a complex number, leaving its phase unchanged.
    """
    abs_z = tf.math.abs(z)
    return tf.cast(tf.keras.activations.relu(abs_z + b), dtype=z.dtype) * z / tf.cast(abs_z + c, dtype=z.dtype)


def crelu(z: Tensor, alpha: float = 0.0, max_value: Optional[float] = None, threshold: float = 0) -> Tensor:
    """
    crelu Function
    --------------
    - Description: This function serves as a reflection of the 'cart_relu' function.
    - Parameters:
    - z: A tensor representing the input.
    - alpha: A float specifying the negative slope of the ReLU (default is 0.0).
    - max_value: An optional float that sets the saturation threshold for the ReLU function. If None, no saturation is applied.
    - threshold: A float that sets the threshold value for the ReLU function (default is 0).
    - Returns: A tensor that is the result of applying the 'cart_relu' function to the input 'z' with specified parameters.
    - Usage: Useful in scenarios where the behavior of 'cart_relu' needs to be replicated or mirrored with specific parameters.
    """
    return cart_relu(z, alpha, max_value, threshold)
    

def cast_to_real(z: Tensor) -> Tensor:
    """
    cast_to_real Function
    ---------------------
    - Description: Converts a complex tensor to its real data type equivalent.
    - Parameters:
    - z: A TensorFlow tensor of complex numbers.
    - Returns: A tensor with the same values as 'z' but cast to the real part of its original data type.
    - Usage: Ideal for operations that require the tensor to be in a real data type, especially in complex number processing.
    """
    return tf.cast(z, z.dtype.real_dtype)


def sigmoid_real(z: Tensor) -> Tensor:
    """
    sigmoid_real Function
    ---------------------
    - Description: Applies the sigmoid activation function to the sum of real and imaginary parts of a complex tensor.
    - Parameters:
    - z: A TensorFlow tensor of complex numbers.
    - Returns: A real-valued tensor resulting from the sigmoid activation of the sum of real and imaginary parts of 'z'.
    - Usage: Useful in neural network layers where a sigmoid activation is needed for complex-valued inputs, combining both real and imaginary parts.
    """
    return tf.keras.activations.sigmoid(tf.math.real(z) + tf.math.imag(z))


def modulus_softmax(z: Tensor, axis=-1) -> Tensor:
    """
    modulus_softmax Function (TYPE A: Cartesian Form)
    -------------------------------------------------
    - Description: Applies softmax to the modulus of a complex input tensor 'z', normalizing it into a probability distribution. Directly applies softmax for real inputs.
    - Softmax Activation: Ensures output values are normalized to the range (0, 1) and sum to 1, ideal for probability distributions.
    - Typical Use: Commonly used in the last layer of classification networks.
    - Calculation: Softmax is calculated as exp(x) / tf.reduce_sum(exp(x)).
    - Reference: TensorFlow documentation - https://www.tensorflow.org/api_docs/python/tf/keras/activations/softmax
    - Parameters:
      - z: Input tensor (complex or real).
      - axis: Axis along which the softmax is applied.
    - Returns: A real-valued tensor resulting from applying softmax to the modulus of 'z' for complex inputs, or directly to 'z' for real inputs.
    """
    return tf.keras.activations.softmax(tf.math.abs(z), axis) if z.dtype.is_complex else tf.keras.activations.softmax(z, axis)


def average_softmax_real_imag(z: Tensor, axis=-1) -> Tensor:
    """
    average_softmax_real_imag Function (TYPE A: Cartesian Form)
    -----------------------------------------------------------
    - Description: Applies softmax separately to the real and imaginary parts of a complex input 'z' and averages the results. Applies softmax directly for real inputs.
    - Softmax Activation: Normalizes outputs to a range of (0, 1), summing to 1.
    - Application: Typically used in the final layer of classification networks for generating probability distributions.
    - Calculation: Softmax is calculated as exp(x) / tf.reduce_sum(exp(x)).
    - Reference: TensorFlow documentation - https://www.tensorflow.org/api_docs/python/tf/keras/activations/softmax
    - Parameters:
      - z: Input tensor (complex or real).
      - axis: Axis for softmax application.
    - Returns: A real-valued tensor from the averaged softmax activations of the real and imaginary parts of 'z'.
    """
    return (0.5 * (tf.keras.activations.softmax(tf.math.real(z), axis) + tf.keras.activations.softmax(tf.math.imag(z), axis))
            if z.dtype.is_complex else tf.keras.activations.softmax(z, axis))


def product_softmax_real_imag(z: Tensor, axis=-1) -> Tensor:
    """
    product_softmax_real_imag Function (TYPE A: Cartesian Form)
    -----------------------------------------------------------
    - Description: Applies softmax to both the real and imaginary parts of complex 'z' and multiplies the results. Applies softmax directly for real tensors.
    - Softmax Activation: Normalizes outputs to the range (0, 1), summing to 1.
    - Role in Networks: Used in the last layer of classification networks for generating probability distributions.
    - Calculation: Softmax is calculated as exp(x) / tf.reduce_sum(exp(x)).
    - Reference: TensorFlow documentation - https://www.tensorflow.org/api_docs/python/tf/keras/activations/softmax
    - Parameters:
      - z: Input tensor (complex or real).
      - axis: Axis for softmax application.
    - Returns: A real-valued tensor from the product of softmax activations of the real and imaginary parts of 'z'.
    """
    return (tf.keras.activations.softmax(tf.math.real(z), axis) * tf.keras.activations.softmax(tf.math.imag(z), axis)
            if z.dtype.is_complex else tf.keras.activations.softmax(z, axis))

    
def absolute_to_real(z: Tensor) -> Tensor:
    """
    absolute_to_real Function
    --------------------------------------------------
    - Description: Takes a complex input tensor 'z' and applies the absolute value operation, resulting in a real-valued output. For real inputs, it returns the input as is.
    - Parameters:
      - z: Input tensor (complex or real).
    - Returns: A real-valued tensor, which is the absolute value of the complex input or the input itself if it's already real.
    """
    return tf.math.abs(z) if z.dtype.is_complex else z


def polar_softmax(z: Tensor, axis=-1) -> Tensor:
    """
    polar_softmax Function (TYPE A: Cartesian Form)
    -----------------------------------------------
    - Description: Applies the softmax function to a complex input tensor 'z' by separately processing its modulus and angle. For real tensors, it applies softmax directly.
    - Softmax Activation: Normalizes output values to a range of (0, 1) with a sum of 1, suitable for probability distribution interpretation.
    - Reference: TensorFlow's documentation on softmax - https://www.tensorflow.org/api_docs/python/tf/keras/activations/softmax
    - Parameters:
      - z: Input tensor (complex or real).
      - axis: The axis along which the softmax normalization is applied.
    - Returns: A tensor resulting from the combined softmax operations on the modulus and angle of 'z' for complex inputs, or a direct softmax application for real inputs.
    """
    if z.dtype.is_complex:
        return 0.5 * (tf.keras.activations.softmax(tf.math.abs(z), axis) + tf.keras.activations.softmax(tf.math.angle(z), axis))
    else:
        return tf.keras.activations.softmax(z, axis)

"""
Cartesian format
"""

def complex_elu(z: Tensor, alpha=1.0) -> Tensor:
    """
    complex_elu Function
    --------------------
    - Description: Applies the Exponential Linear Unit (ELU) activation function to both the real and imaginary parts of a complex input tensor 'z'. The ELU is defined as x for x > 0 and alpha * (exp(x) - 1) for x < 0.
    - Reference: Learn more about ELU from TensorFlow's documentation - https://www.tensorflow.org/api_docs/python/tf/keras/activations/elu
    - Parameters:
      - z: Input tensor (complex).
      - alpha: A scalar representing the slope of the negative section.
    - Returns: A complex tensor resulting from applying the ELU activation function to both the real and imaginary parts of 'z'.
    """
    real_part = tf.keras.activations.elu(tf.math.real(z), alpha)
    imag_part = tf.keras.activations.elu(tf.math.imag(z), alpha)
    return tf.cast(tf.complex(real_part, imag_part), dtype=z.dtype)


def complex_exponential(z: Tensor) -> Tensor:
    """
    complex_exponential Function
    ----------------------------
    - Description: Applies the exponential activation function, exp(x), to both the real and imaginary parts of a complex input tensor 'z'.
    - Reference: More details on the exponential activation can be found at TensorFlow's documentation - https://www.tensorflow.org/api_docs/python/tf/keras/activations/exponential
    - Parameters:
      - z: Input tensor (complex).
    - Returns: A complex tensor resulting from applying the exponential activation function to both the real and imaginary parts of 'z'.
    """
    real_part = tf.keras.activations.exponential(tf.math.real(z))
    imag_part = tf.keras.activations.exponential(tf.math.imag(z))
    return tf.cast(tf.complex(real_part, imag_part), dtype=z.dtype)


def cart_relu(z: Tensor, alpha: float = 0.0, max_value: Optional[float] = None, threshold: float = 0) -> Tensor:
    """
    Applies Rectified Linear Unit to both the real and imag part of z
    The relu function, with default values, it returns element-wise max(x, 0).
    Otherwise, it follows:  f(x) = max_value for x >= max_value,
                            f(x) = x for threshold <= x < max_value,
                            f(x) = alpha * (x - threshold) otherwise.
    https://www.tensorflow.org/api_docs/python/tf/keras/activations/relu
    :param z: Tensor -- Input tensor.
    :param alpha: float -- A float that governs the slope for values lower than the threshold (default 0.0).
    :param max_value: Optional float -- A float that sets the saturation threshold (the largest value the function will return)
        (default None).
    :param threshold: float -- A float giving the threshold value of the activation function below which
        values will be damped or set to zero (default 0).
    :return: Tensor result of the applied activation function
    """
    return tf.cast(tf.complex(tf.keras.activations.relu(tf.math.real(z), alpha, max_value, threshold),
                              tf.keras.activations.relu(tf.math.imag(z), alpha, max_value, threshold)), dtype=z.dtype)


def complex_leaky_relu(z: Tensor, alpha=0.2, name=None) -> Tensor:
    """
    complex_leaky_relu Function
    ---------------------------
    - Description: Applies the Leaky Rectified Linear Unit (Leaky ReLU) activation function separately to both the real and imaginary parts of a complex input tensor 'z'.
    - Reference: For more information, see TensorFlow's documentation on Leaky ReLU - https://www.tensorflow.org/api_docs/python/tf/nn/leaky_relu and the related academic paper at http://robotics.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf
    - Parameters:
      - z: Input tensor (complex).
      - alpha: Slope of the activation function for x < 0. Default value is 0.2.
      - name: Optional name for the operation.
    - Returns: A complex tensor resulting from applying the Leaky ReLU activation to both the real and imaginary parts of 'z'.
    """
    real_part = tf.nn.leaky_relu(tf.math.real(z), alpha, name)
    imag_part = tf.nn.leaky_relu(tf.math.imag(z), alpha, name)
    return tf.cast(tf.complex(real_part, imag_part), dtype=z.dtype)


def complex_softsign(z: Tensor) -> Tensor:
    """
    complex_softsign Function
    -------------------------
    - Description: Applies the Softsign activation function separately to both the real and imaginary parts of a complex input tensor 'z'.
    - Softsign Activation: Described by the formula x / (abs(x) + 1).
    - Reference: TensorFlow's documentation on Softsign - https://www.tensorflow.org/api_docs/python/tf/keras/activations/softsign (Note: There's a typo in the TensorFlow references, mentioning 'softplus' instead of 'softsign').
    - Parameters:
      - z: Input tensor (complex).
    - Returns: A complex tensor resulting from applying the Softsign activation to both the real and imaginary parts of 'z'.
    """
    real_part = tf.keras.activations.softsign(tf.math.real(z))
    imag_part = tf.keras.activations.softsign(tf.math.imag(z))
    return tf.cast(tf.complex(real_part, imag_part), dtype=z.dtype)
