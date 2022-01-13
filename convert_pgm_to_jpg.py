import os
from PIL import Image

for folder in os.listdir('face-database/archive'):
    #print(folder)
    for file in os.listdir(f'face-database/archive/{folder}'):
        #rint(file)
        #'''
        filename, extension  = os.path.splitext(file)
        if extension == ".pgm":
            new_file = "{}.jpg".format(filename)
            with Image.open(f'face-database/archive/{folder}/{file}') as im:
                im.save(f'face-database/data/{folder}_{new_file}')
        #'''