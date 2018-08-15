from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

cpredictor=load_model('alphabet_recognition_model.h5')


test_image = image.load_img(r'/home/rishabh/Downloads/alphabet_recognition/Testing/f/24.png', target_size = (32, 32))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cpredictor.predict(test_image)

if result[0][0] == 1:
    print("Letter is a")
elif result[0][1] == 1:
    print("Letter is b")
elif result[0][2] == 1:
    print("Letter is c")
elif result[0][3] == 1:
    print("Letter is d")
elif result[0][4] == 1:
    print("Letter is e")
elif result[0][5] == 1:
    print("Letter is f")
elif result[0][6] == 1:
    print("Letter is g")
elif result[0][7] == 1:
    print("Letter is h")
elif result[0][8] == 1:
    print("Letter is i")
elif result[0][9] == 1:
    print("Letter is j")
elif result[0][10] == 1:
    print("Letter is k")
elif result[0][11] == 1:
    print("Letter is l")
elif result[0][12] == 1:
    print("Letter is m")
elif result[0][13] == 1:
    print("Letter is n")
elif result[0][14] == 1:
    print("Letter is o")
elif result[0][15] == 1:
    print("Letter is p")
elif result[0][16] == 1:
    print("Letter is q")
elif result[0][17] == 1:
    print("Letter is r")
elif result[0][18] == 1:
    print("Letter is s")
elif result[0][19] == 1:
    print("Letter is t")
elif result[0][20] == 1:
    print("Letter is u")
elif result[0][21] == 1:
    print("Letter is v")
elif result[0][22] == 1:
    print("Letter is w")
elif result[0][23] == 1:
    print("Letter is x")
elif result[0][24] == 1:
    print("Letter is y")
elif result[0][25] == 1:
    print("Letter is z")
else:
    print("No match found")