import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.preprocessing import image

img_path = '/home/artur/python_projects/Machine-sorting-bolts-and-nuts/tests/test1.jpg'
model_path = '/home/artur/python_projects/Machine-sorting-bolts-and-nuts/bolts_and_nuts_small_1.h5'


# Converting colors
img = Image.open(img_path).convert('L')

npim = np.array(img)
res = np.zeros((npim.shape[0], npim.shape[1], 3), dtype=np.uint8)
res[npim < 245] = [150, 150, 150]
Image.fromarray(res).save('result.png')

img_path = 'result.png'

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
plt.title('%.2f percent bolt and %.2f percent nut' % (100 * (1 - score), 100 * score))
plt.show()
