import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
import scipy


import matplotlib.pyplot as plt
import matplotlib.image as mpimg


import zipfile
import os

def unzip():
    '''
    local_zip = './DATA/horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('./DATA/horse-or-human')
    zip_ref.close()
    '''
    local_zip = './DATA/validation-horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('./DATA/validation-horse-or-human')
    zip_ref.close()

unzip()

train_horse_dir = os.path.join('./DATA/horse-or-human/horses')
train_human_dir = os.path.join('./DATA/horse-or-human/humans')
# Directory with validation horse pictures
validation_horse_dir = os.path.join('./DATA/validation-horse-or-human/horses')
validation_human_dir = os.path.join('./DATA/validation-horse-or-human/humans')

print('total training horse imgs:', len(os.listdir(train_horse_dir)))
print('total training human imgs:', len(os.listdir(train_human_dir)))

def testDraw():
    train_horse_names = os.listdir(train_horse_dir)
    print(train_horse_names[:10])

    train_human_names = os.listdir(train_human_dir)
    print(train_human_names[:10])

    # Parameters for our graph; we'll output images in a 4x4 configuration
    nrows = 4
    ncols = 4

    # Index for iterating over images
    pic_index = 0
    fig = plt.gcf()
    fig.set_size_inches(ncols * 4, nrows * 4)

    pic_index += 8
    next_horse_pix = [os.path.join(train_horse_dir, fname)
                      for fname in train_horse_names[pic_index - 8:pic_index]]
    next_human_pix = [os.path.join(train_human_dir, fname)
                      for fname in train_human_names[pic_index - 8:pic_index]]

    for i, img_path in enumerate(next_horse_pix + next_human_pix):
        # Set up subplot; subplot indices start at 1
        sp = plt.subplot(nrows, ncols, i + 1)
        sp.axis('Off')  # Don't show axes (or gridlines)

        img = mpimg.imread(img_path)
        plt.imshow(img)

    plt.show()

model = keras.models.Sequential([
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300,300,3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# model.summary()
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

# DATA PROCESSING
train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    './DATA/horse-or-human',
    target_size=(300, 300),
    batch_size=128,
    class_mode='binary'
)
# Flow validation images in batches of 128 using validation_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        './DATA/validation-horse-or-human/',  # This is the source directory for validation images
        target_size=(300, 300),  # All images will be resized to 300x300
        batch_size=32,
        # Since you use binary_crossentropy loss, you need binary labels
        class_mode='binary')

history = model.fit(
    train_generator,
    steps_per_epoch=8,
    epochs=15,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=8
)