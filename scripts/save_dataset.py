from detection.models import Detection
from pathlib import Path
import os
import shutil


def save_dogs_dataset(folder='media/dataset/dogs/train', split=1.0):
    detections = Detection.objects.filter(animal='dog', rating=1)
    Path(folder).mkdir(parents=True, exist_ok=True)
    for dog in detections:
        source = os.path.join('media',dog.picture.name)
        breed_folder = os.path.join(folder,dog.breed)
        Path(breed_folder).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, os.path.join(breed_folder, source.split('/')[-1]))
        print(f'copied {dog.animal} {source} to breed {dog.breed}')


def save_cats_vs_dogs_dataset(folder='media/dataset/animals/train', split=1.0):
    detections = Detection.objects.filter(rating=1)
    cats_folder = os.path.join(folder,'cat')
    dogs_folder = os.path.join(folder,'dog')
    
    Path(folder).mkdir(parents=True, exist_ok=True)
    Path(cats_folder).mkdir(parents=True, exist_ok=True)
    Path(dogs_folder).mkdir(parents=True, exist_ok=True)

    for detection in detections:
        source = os.path.join('media',detection.picture.name)
        if detection.animal == 'cat':
            shutil.copyfile(source, os.path.join(cats_folder,source.split('/')[-1]))
        elif detection.animal == 'dog':
            shutil.copyfile(source, os.path.join(dogs_folder,source.split('/')[-1]))


def save_dataset(folder='media/dataset'):
    detections = Detection.objects.all()
    print(folder)
    cats_folder = os.path.join(folder,'cat')
    dogs_folder = os.path.join(folder,'dog')
    # create folders for dataset
    Path(folder).mkdir(parents=True, exist_ok=True)
    Path(cats_folder).mkdir(parents=True, exist_ok=True)
    Path(dogs_folder).mkdir(parents=True, exist_ok=True)

    for det in detections:
        if det.rating == 1:
            # if cat, put it in cats folder
            source = os.path.join('media',det.picture.name)
            if det.animal == 'cat':
                shutil.copyfile(source, os.path.join(cats_folder,source.split('/')[-1]))
                print(f'copied {det.animal} {source}')
            else:
                # detect breed and create folder if not exists
                breed_folder = os.path.join(dogs_folder,det.breed)
                Path(breed_folder).mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, os.path.join(breed_folder, source.split('/')[-1]))
                print(f'copied {det.animal} {source} to breed {det.breed}')


def run(*args):
    print('creando dataset...')
    if len(args) == 1:
        if args[0] == 'catsvdogs':
            save_cats_vs_dogs_dataset()
        elif args[0] == 'dogs':
            save_dogs_dataset()
    else:
        print('ERROR: correct usage: python manage.py runscript save_dataset --script-args [catsvdogs, dogs]')
