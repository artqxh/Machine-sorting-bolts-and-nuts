import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.preprocessing import image

img_path = '/home/artur/python_projects/Machine-sorting-bolts-and-nuts/tests/bolt.201.png'
model_path = '/home/artur/python_projects/Machine-sorting-bolts-and-nuts/bolts_and_nuts_small_1.h5'

model = keras.models.load_model(model_path)
image_size = (150, 150)

img = keras.preprocessing.image.load_img(img_path, target_size=image_size)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = predictions[0]
print(model.summary(), 'This image is %.2f percent bolt and %.2f percent nut.' % (100 * (1 - score), 100 * score))

img = image.load_img(img_path, target_size=image_size)
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255

plt.imshow(img_tensor[0])
plt.title('%.2f percent bolt and %.2f percent nut' % (100 * (1 - score), 100 * score))
plt.show()
