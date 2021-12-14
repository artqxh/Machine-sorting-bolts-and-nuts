import os, shutil

# Remove folder
shutil.rmtree('/home/artur/python_projects/Machine-sorting-bolts-and-nuts/data/bolts_and_nuts_small')

# Copying images to sets and subsets
original_dataset_dir = '/home/artur/python_projects/Machine-sorting-bolts-and-nuts/data/original_data'

base_dir = '/home/artur/python_projects/Machine-sorting-bolts-and-nuts/data/bolts_and_nuts_small'
os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

train_bolts_dir = os.path.join(train_dir, 'bolts')
os.mkdir(train_bolts_dir)
validation_bolts_dir = os.path.join(validation_dir, 'bolts')
os.mkdir(validation_bolts_dir)
test_bolts_dir = os.path.join(test_dir, 'bolts')
os.mkdir(test_bolts_dir)

train_nuts_dir = os.path.join(train_dir, 'nuts')
os.mkdir(train_nuts_dir)
validation_nuts_dir = os.path.join(validation_dir, 'nuts')
os.mkdir(validation_nuts_dir)
test_nuts_dir = os.path.join(test_dir, 'nuts')
os.mkdir(test_nuts_dir)

fnames = ['bolt.{}.png'.format(i) for i in range(100)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_bolts_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['bolt.{}.png'.format(i) for i in range(100, 150)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_bolts_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['bolt.{}.png'.format(i) for i in range(150, 200)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_bolts_dir, fname)
    shutil.copyfile(src, dst)


fnames = ['nut.{}.png'.format(i) for i in range(100)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_nuts_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['nut.{}.png'.format(i) for i in range(100, 150)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_nuts_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['nut.{}.png'.format(i) for i in range(150, 200)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_nuts_dir, fname)
    shutil.copyfile(src, dst)


# Building a neural network
from keras import layers, models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# Trained model configuration
from tensorflow.keras import optimizers

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])


# Loading images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=10, class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150, 150), batch_size=10, class_mode='binary')


# Model fitting
history = model.fit_generator(train_generator, steps_per_epoch=3, epochs=30, validation_data=validation_generator, validation_steps=3)


# Save the model
model.save('bolts_and_nuts_small_1.h5')


# Data visualization
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Dokładność trenowania')
plt.plot(epochs, val_acc, 'r', label='Dokładność walidacji')
plt.title('Dokładność trenowania i walidacji')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Strata trenowania')
plt.plot(epochs, val_loss, 'r', label='Strata walidacji')
plt.title('Strata trenowania i walidacji')
plt.legend()

plt.show()




