import os

from PIL import Image

for filename in os.listdir('../IMAGES'):
    im = Image.open(r'../IMAGES/' + filename)
    im.save(r'../IMAGES/' + filename[:-3] + 'png')