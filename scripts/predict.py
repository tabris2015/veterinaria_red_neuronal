import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from utils import build_model, predict_image
from PIL import Image


# archivo a predecir
file = '../data/images/perro1.jpg'

# obtener etiquetas
_, breed_info = tfds.load(
    'stanford_dogs', with_info=True, as_supervised=True
)

_, cats_dogs_info = tfds.load(
    'cats_vs_dogs', with_info=True, as_supervised=True
)


# print(f'cats_vs_dogs: {cats_dogs_info.features["label"].num_classes}')
# print(f'dog breeds: {breed_info.features["label"].num_classes}')

# cargar modelos
dogs_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

cats_dogs_model = build_model(num_classes=2)
cats_dogs_model.load_weights('../tf_models/colab/cats_dogs_best_colab1.h5')

dogs_model = tf.keras.models.load_model('../tf_models/colab/best_colab1_fine.h5', compile=False)
dogs_model.compile(
        optimizer=dogs_optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

# realizar detección gato o perro
res = predict_image(file, cats_dogs_model, cats_dogs_info, show_image=False)

print(f'Animal: {res}')
# si es perro detectar raza
if res == 'dog':
    breed = predict_image(file, dogs_model, breed_info, show_image=False)
    print(f'Raza: {breed}')
