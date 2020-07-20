import pickle
import tensorflow as tf
from tensorboard import summary
from config import Config
import datetime


class Model:
    def __init__(self, ds):
        self.ds = ds
        if Config.MODE == 'train':
            self.create_model()

    def create_model(self):
        mirrored_strategy = tf.distribute.MirroredStrategy()

        with mirrored_strategy.scope():
            base_model = tf.keras.applications.NASNetLarge(include_top=False,
                                                           weights='imagenet',
                                                           classes=39)
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
            prediction_layer_two = tf.keras.layers.Dense(39,
                                                         activation='sigmoid')

            self.model = tf.keras.Sequential([
                base_model, global_average_layer, prediction_layer_one,
                prediction_layer_two
            ])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['binary_accuracy'])

    def fit(self):
        tensorboard_cb = tf.keras.callbacks.TensorBoard(str(Config.LOG_DIR))
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(verbose=1,
                                                             patience=30,
                                                             min_delta=1)
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            f'{Config.CHECKPOINT_DIR}/best_model.h5',
            save_best=True,
            save_weights_only=True)
        lr_scheduler_cb = tf.keras.callbacks.ReduceLROnPlateau(factor=0.2,
                                                               min_lr=0.001)
        history = self.model.fit(self.ds.train,
                                 steps_per_epoch=35987 // Config.BATCH_SIZE,
                                 epochs=Config.EPOCHS,
                                 validation_data=self.ds.val,
                                 validation_steps=2952 / Config.BATCH_SIZE,
                                 callbacks=[
                                     tensorboard_cb, checkpoint_cb,
                                     early_stopping_cb, lr_scheduler_cb
                                 ])

    def eval(self):
        return self.model.evaluate(self.ds.test,
                                   steps=2953 // Config.BATCH_SIZE)
