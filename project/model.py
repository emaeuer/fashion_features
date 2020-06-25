import tensorflow as tf
from tensorboard import summary
from config import Config
import datetime


class Model:
    def __init__(self, ds):
        self.ds = ds
        self.create_model()

    def create_model(self):
        mirrored_strategy = tf.distribute.MirroredStrategy()
        
        with mirrored_strategy.scope(): 
            base_model = tf.keras.applications.NASNetLarge(include_top=False,
                                                       weights='imagenet',classes=39)
            base_model.trainable = True
            set_trainable = False
            for layer in base_model.layers:
                if layer.name == 'activation_166':
                    set_trainable = True
                if set_trainable:
                   layer.trainable = True
                else:
                   layer.trainable = False
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
        #log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

        self.model.fit(self.ds.train,
                       steps_per_epoch=44199 // Config.BATCH_SIZE,
                       epochs=1,
                       validation_data=self.ds.val,
                       validation_steps=4444 // Config.BATCH_SIZE,
                       callbacks=[])

    def eval(self):
        return self.model.evaluate(self.ds.test,
                                   steps=4445 // Config.BATCH_SIZE)
