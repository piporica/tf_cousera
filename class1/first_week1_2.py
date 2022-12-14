import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

def myFirstModel():
    model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')

    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

    model.fit(xs, ys, epochs=500)
    print(model.predict([10.0, 11.0]))


def useDataSet():
    callbacks = myCallback()
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.fit(train_images, train_labels, epochs=5, callbacks=[callbacks])
    model.evaluate(test_images, test_labels)

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss') <0.2):
            print('\nLoss is low so cancelling training!')
            self.model.stop_training = True

useDataSet()