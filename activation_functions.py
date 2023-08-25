import tensorflow as tf


def modrelu(z: tf.Tensor, b: float = 1., c: float = 1e-3) -> tf.Tensor:
    abs_z = tf.math.abs(z)
    return tf.cast(tf.keras.activations.relu(abs_z + b), dtype=z.dtype) * z / tf.cast(abs_z + c, dtype=z.dtype)


def complex_leaky_relu(z: tf.Tensor, alpha: float = 0.3):
    real = tf.math.real(z)
    imag = tf.math.imag(z)
    return tf.complex(tf.keras.layers.LeakyReLU(alpha)(real), tf.keras.layers.LeakyReLU(alpha)(imag))


def complex_cardioid(z: tf.Tensor) -> tf.Tensor:
    return tf.cast(1 + tf.math.cos(tf.math.angle(z)), dtype=z.dtype) * z / 2.
