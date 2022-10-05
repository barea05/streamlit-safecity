import tensorflow as tf

model_path = "./data/keras/"
keras_model = tf.keras.models.load_model(model_path)

#load xg_boost

def get_keras_model():
    return keras_model


def get_xg_boost():
    return None
