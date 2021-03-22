#%%
import os
import shutil
import numpy as np
import pandas as pd
labels_csv = pd.read_csv('data/labels.csv')
# %%
train_path = "data/train/"

filenames = labels_csv['id'].apply(lambda id: train_path + id + '.jpg')

labels_csv['file'] = filenames

# %%
unique_breeds = np.unique(labels_csv['breed'])
# %%
# create folders
for breed in unique_breeds:
    breed_path = train_path + breed
    try:
        os.mkdir(breed_path)
    except OSError:
        print("Creation of the directory %s failed" % breed_path)
    else:
        print("Successfully created the directory %s " % breed_path)

#%%
# move images to folder
import shutil
for i, row in labels_csv.iterrows():
    source = row['file']
    destination = train_path + row['breed'] + '/' + row['id'] + '.jpg'
    res = shutil.move(source, destination)
    
# %%
