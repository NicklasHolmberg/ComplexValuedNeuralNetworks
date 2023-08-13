import numpy as np
from activation_functions import complex_cardioid, complex_leaky_relu, complex_tanh, zrelu, modrelu
import tensorflow as tf
from cvnn.layers import ComplexDense, ComplexFlatten
import constants

def cast_to_real(x):
    return tf.cast(x, tf.float32)


def create_model(activation_function):
    if activation_function in [modrelu, complex_leaky_relu, complex_tanh, complex_cardioid, zrelu]:
        model = tf.keras.models.Sequential([
            ComplexFlatten(input_shape=(28, 28, 1), dtype=np.complex64),
            ComplexDense(128, activation=activation_function, dtype=np.complex64),
            ComplexDense(10, activation=cast_to_real, dtype=np.complex64),
            tf.keras.layers.Activation('softmax')
        ])
    else:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
            tf.keras.layers.Dense(128, activation=activation_function),
            tf.keras.layers.Dense(constants.NUM_CLASSES, activation='softmax')
        ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
