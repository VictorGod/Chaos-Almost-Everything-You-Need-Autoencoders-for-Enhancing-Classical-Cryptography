import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Кастомный слой для регуляризации латентного пространства
class VarianceRegularizer(layers.Layer):
    def __init__(self, lambda_reg=0.01, **kwargs):
        super(VarianceRegularizer, self).__init__(**kwargs)
        self.lambda_reg = lambda_reg

    def call(self, inputs):
        # Штраф за низкую дисперсию по каждому измерению
        variance_loss = -self.lambda_reg * tf.reduce_mean(tf.math.reduce_variance(inputs, axis=0))
        self.add_loss(variance_loss)
        return inputs

# Составная функция активации для усиления нелинейностей в латентном пространстве
def chaos_activation(x):
    return tf.sin(8.0 * x) + 0.5 * tf.tanh(4.0 * x)

def build_autoencoder(image_size=(28, 28)):
    input_img = keras.Input(shape=(*image_size, 1))
    x = layers.Flatten()(input_img)
    
    # Энкодер: применяем Dense-слой и нелинейную активацию
    x = layers.Dense(128)(x)
    x = layers.Activation(chaos_activation)(x)
    
    # Латентное представление с дополнительной регуляризацией
    latent = layers.Dense(64, name="latent")(x)
    latent = layers.Activation(chaos_activation)(latent)
    latent = VarianceRegularizer(lambda_reg=0.01)(latent)
    
    # Декодер: восстанавливаем изображение с использованием BatchNormalization
    x = layers.Dense(128)(latent)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(chaos_activation)(x)
    
    # Выходной слой для восстановления изображения в диапазоне [0,1]
    decoded = layers.Dense(np.prod(image_size), activation='sigmoid')(x)
    decoded = layers.Reshape((*image_size, 1))(decoded)
    
    autoencoder = keras.Model(input_img, decoded)
    encoder = keras.Model(input_img, latent)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder
