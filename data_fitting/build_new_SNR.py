import os

from PIL import Image
import random


def randomize_pixels(image, probability):
    pixels = image.load()
    width, height = image.size
    for x in range(width):
        for y in range(height):
            if random.random() < probability:
                pixels[x, y] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


for filename in os.listdir('diff_SNR/test20'):
    a = os.path.join('diff_SNR/test20', filename)
    for file in os.listdir(a):
        file_path = os.path.join(a, file)
        image = Image.open(file_path)
        image = image.convert('RGB')

        randomize_pixels(image, probability=0.2)
        image.save(file_path)
