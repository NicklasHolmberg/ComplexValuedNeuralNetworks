import tensorflow as tf


def modrelu(z: tf.Tensor, b: float = 1., c: float = 1e-3) -> tf.Tensor:
    abs_z = tf.math.abs(z)
    return tf.cast(tf.keras.activations.relu(abs_z + b), dtype=z.dtype) * z / tf.cast(abs_z + c, dtype=z.dtype)


def complex_leaky_relu(z: tf.Tensor, alpha: float = 0.3):
    real = tf.math.real(z)
    imag = tf.math.imag(z)
    return tf.complex(tf.keras.layers.LeakyReLU(alpha)(real), tf.keras.layers.LeakyReLU(alpha)(imag))

def complex_tanh(z: tf.Tensor):
    return tf.math.tanh(z)


def complex_cardioid(z: tf.Tensor) -> tf.Tensor:
    return tf.cast(1 + tf.math.cos(tf.math.angle(z)), dtype=z.dtype) * z / 2.

def zrelu(z: tf.Tensor, epsilon=1e-7) -> tf.Tensor:
    imag_relu = tf.nn.relu(tf.math.imag(z))
    real_relu = tf.nn.relu(tf.math.real(z))
    ret_real = imag_relu * real_relu / (imag_relu + epsilon)
    ret_imag = imag_relu * real_relu / (real_relu + epsilon)
    return tf.complex(ret_real, ret_imag)