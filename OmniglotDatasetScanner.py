import os

import numpy
from PIL import Image

data = "images_all"
imglist=[]

#TODO: use img.transpose methods (rotation & mirroring) to augment dataset

for alphabet in os.scandir(data):
    print(alphabet.name)
    for character in os.scandir(alphabet.path):
        print(character.name)
        for image in os.scandir(character.path):
            imgfull = Image.open(image.path)
            im = imgfull.crop((4,4,100,100))
            temparray = numpy.array(im.getdata(), numpy.uint8).reshape(im.size[1], im.size[0])
            temparray =temparray.astype(numpy.float32)/255
            print(temparray.shape)
            imglist.append(temparray)

all_dataset = numpy.array(imglist)

numpy.save("images_all_processed", all_dataset)
