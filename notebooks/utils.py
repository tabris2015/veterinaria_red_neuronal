# build model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from PIL import Image

IMG_SIZE = 224
batch_size = 64

def format_label(label, ds_info):
    text = ds_info.features["label"].int2str(label)
    return text if len(text.split('-')) < 2 else (' ').join(text.split('-')[1:])

def predict_image(file, model, ds_info, show_image=True):
    img = np.array(Image.open(file).resize((224,224)), dtype=np.float32) #/ 255.0
    pred = model.predict(img.reshape(-1, 224, 224, 3))
    if ds_info.features['label'].num_classes > 3:
        top3 = np.argsort(pred, axis=1)[0, -3:]
        for label in top3:
            print(f'{format_label(label, ds_info)}: {pred[0,label]}')
    pred_label = format_label(np.argmax(pred), ds_info)

    if show_image:
        plt.imshow(img.astype("uint8"))
        plt.title("{}".format(format_label(np.argmax(pred), ds_info)))
    return pred_label

img_augmentation = Sequential(
    [
        preprocessing.RandomRotation(factor=0.15),
        preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
        preprocessing.RandomFlip(),
        preprocessing.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)

def build_model(num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = img_augmentation(inputs)
    model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(2, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model
