import cv2
import time
import serial
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from tensorflow import keras
from keras.preprocessing import image


# Taking photo from camera
cap = cv2.VideoCapture(-1)
ret, frame = cap.read()

def takePicture():
    (grabbed, frame) = cap.read()
    showimg = frame
    cv2.waitKey(1)
    image = 'result.png'
    cv2.imwrite(image, frame)
    cap.release()
    return image

print(takePicture())


# Zooming
img = Image.open("result.png")

def zoom_at(img, x, y, zoom):
    w, h = img.size
    zoom2 = zoom * 2
    img = img.crop((x - w / zoom2, y - h / zoom2,
                    x + w / zoom2, y + h / zoom2))
    return img.resize((w, h), Image.LANCZOS)

img = zoom_at(img, 345, 245, 3.5)
img = img.save('result.png')


# Reading image and model
img_path = '/home/artur/python_projects/Machine-sorting-bolts-and-nuts/tests/8.png'
model_path = '/home/artur/python_projects/Machine-sorting-bolts-and-nuts/bolts_and_nuts_small_1.h5'
imgplot = plt.imshow(mpimg.imread(img_path))
plt.show()


# Binarization
originalImage = cv2.imread(img_path)
grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
(thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)

#originalImage, grayImage, blackAndWhiteImage
cv2.imwrite('result.png', blackAndWhiteImage)
img_path = '/home/artur/python_projects/Machine-sorting-bolts-and-nuts/result.png'


# Loading model
model = keras.models.load_model(model_path)
img_size = (150, 150)

img = keras.preprocessing.image.load_img(img_path, target_size=img_size)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = predictions[0]
print(model.summary())

img = image.load_img(img_path, target_size=img_size)
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255


# Computing values
bolt = 100 * (1 - score)
nut = 100 * score

if bolt > nut:
    value = str(0)
    message = 'śrubka'

elif bolt <= nut:
    value = str(1)
    message = 'nakrętka'

plt.imshow(img_tensor[0])
plt.title('To jest najprawdopodobniej {}'.format(message))
plt.show()


# Sending signal to arduino
port = '/dev/ttyACM0'
data = serial.Serial(port, baudrate=9600, bytesize=serial.EIGHTBITS)

time.sleep(1)
data.write(value.encode())
data.close()
