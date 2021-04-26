import numpy as np
import tensorflow as tf
from PIL import Image
from .utils import build_model, format_label


class Predictor(object):
    IMG_SIZE = 224
    BREED_FILE = 'detection/breeds.txt'
    ANIMAL_FILE = 'detection/animals.txt'

    def __init__(self, cats_dogs_file, dogs_file):
        self.dogs_file = dogs_file
        self.cats_dogs_file = cats_dogs_file
        # obtener etiquetas
        self.animal_labels = []
        with open(self.ANIMAL_FILE, 'r') as f:
            for line in f:
                self.animal_labels.append(line.strip())

        self.breed_labels = []
        with open(self.BREED_FILE, 'r') as f:
            for line in f:
                self.breed_labels.append(line.strip())

        # cargar modelos
        dogs_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

        self.cats_dogs_model = build_model(num_classes=2)
        self.cats_dogs_model.load_weights(self.cats_dogs_file)

        self.dogs_model = tf.keras.models.load_model(self.dogs_file, compile=False)
        self.dogs_model.compile(
            optimizer=dogs_optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
        )
        print('iniciado Sercicio de prediccion')

    def predict_file(self, file, model, ds_info):
        img = np.array(Image.open(file).resize((self.IMG_SIZE, self.IMG_SIZE)), dtype=np.float32)
        pred = model.predict(img.reshape(-1, 224, 224, 3))
        if ds_info.features['label'].num_classes > 3:
            top3 = np.argsort(pred, axis=1)[0, -3:]
            for label in top3:
                print(f'{format_label(label, ds_info)}: {pred[0, label]}')
        pred_label = format_label(np.argmax(pred), ds_info)
        return pred_label

    def predict_img(self, img, model, labels, return_dict=False):
        img = img.resize((self.IMG_SIZE, self.IMG_SIZE))
        img = img.convert('RGB')
        img_array = np.asarray(img, dtype=np.float32)[:, :, :3]
        print(f'received size: {img.size} final size: {img_array.shape}')
        # prediction
        pred = model.predict(img_array.reshape(-1, 224, 224, 3))
        res_dict = {}
        if len(labels) > 3:
            top3 = np.flip(np.argsort(pred, axis=1)[0, -3:])
            for i, label in enumerate(top3):
                text = labels[label]
                confidence = pred[0, label]
                res_dict[str(i)] = text + ': ' + str(confidence)
                print(f'{text}: {confidence}')

        pred_label = labels[np.argmax(pred)]
        if return_dict:
            res_dict['main'] = pred_label
            return res_dict
        return pred_label

    def predict(self, img):
        res = self.predict_img(img, self.cats_dogs_model, self.animal_labels)
        breed = {'main': 'gato '}
        if res == 'perro':
            breed = self.predict_img(img, self.dogs_model, self.breed_labels, return_dict=True)
        return {'animal': res, 'raza': breed}
