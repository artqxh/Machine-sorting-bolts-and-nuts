import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow import keras
from keras.preprocessing import image


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


img = Image.open("result.png")

def zoom_at(img, x, y, zoom):
    w, h = img.size
    zoom2 = zoom * 2
    img = img.crop((x - w / zoom2, y - h / zoom2,
                    x + w / zoom2, y + h / zoom2))
    return img.resize((w, h), Image.LANCZOS)

img = zoom_at(img, 345, 245, 3.5)
img = img.save('result.png')


img_path = '/home/artur/python_projects/Machine-sorting-bolts-and-nuts/results.png'
model_path = '/home/artur/python_projects/Machine-sorting-bolts-and-nuts/bolts_and_nuts_small_1.h5'


model = keras.models.load_model(model_path)
img_size = (150, 150)

img = keras.preprocessing.image.load_img(img_path, target_size=img_size)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = predictions[0]
print(model.summary(), 'This image is %.2f percent bolt and %.2f percent nut.' % (100 * (1 - score), 100 * score))

img = image.load_img(img_path, target_size=img_size)
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255

plt.imshow(img_tensor[0])
bolt = 100 * (1 - score)
nut = 100 * score
print(bolt, nut)


plt.title('%.2f percent bolt and %.2f percent nut' % (100 * (1 - score), 100 * score))
plt.show()
