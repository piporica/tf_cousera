import numpy as np
import tensorflow as tf
from tensorflow import keras

def reshape_and_moramlize(images):
    images = np.reshape(images, (*images.shape, 1))
    images = images / 255.0
    return images

class myCallback(tf.keras.callbacks.Callback):
    # Define the method that checks the accuracy at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') is not None and logs.get('accuracy') > 0.995:
            print('\nachive 99.5 accuracy! so stop training')
            self.model.stop_training = True


def convolutional_model():
    # Define the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = reshape_and_moramlize(train_images)
test_images = reshape_and_moramlize(test_images)
model = convolutional_model()
callbacks = myCallback()
model.fit(train_images, train_labels, epochs=10, callbacks=[callbacks])
model.evaluate(train_images, train_labels)
