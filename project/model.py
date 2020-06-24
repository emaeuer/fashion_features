import tensorflow as tf
from tensorboard import summary
from config import Config
import datetime


class Model:
    def __init__(self, ds):
        self.ds = ds
        self.create_model()

    def create_model(self):
        base_model = tf.keras.applications.NASNetLarge(include_top=False,
                                                       weights='imagenet')
        base_model.trainable = False
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        prediction_layer_one = tf.keras.layers.Dense(128)
        prediction_layer_two = tf.keras.layers.Dense(39, activation='sigmoid')

        self.model = tf.keras.Sequential([
            base_model, global_average_layer, prediction_layer_one,
            prediction_layer_two
        ])

        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss=tf.keras.losses.BinaryCrossentropy(),
                           metrics=['accuracy'])

    def fit(self):
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

        self.model.fit(self.ds.train,
                       steps_per_epoch=44199 // Config.BATCH_SIZE,
                       epochs=1,
                       validation_data=self.ds.val,
                       validation_steps=4444 // Config.BATCH_SIZE,
                       callbacks=[tensorboard_callback])

    def eval(self):
        return self.model.evaluate(self.ds.test,
                                   steps=4445 // Config.BATCH_SIZE)
