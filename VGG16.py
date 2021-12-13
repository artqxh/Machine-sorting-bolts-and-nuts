import numpy as np
from keras.preprocessing import image
from tensorflow.keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions

model = VGG16(weights='imagenet')

img_path = '/home/artur/python_projects/Machine-sorting-bolts-and-nuts/image6.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])