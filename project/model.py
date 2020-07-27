import pickle
import tensorflow as tf
from tensorboard import summary
from config import Config
import datetime
import numpy as np


class Model:
    def __init__(self, ds, pre_trained_model):
        self.ds = ds
        if Config.MODE == 'train' or Config.MODE == 'predict_images':
            self.create_model(pre_trained_model)

    # source for loss and metric functions:
    # https://towardsdatascience.com/the-unknown-benefits-of-using-a-soft-f1-loss-in-classification-systems-753902c0105d
    @staticmethod
    def macro_soft_f1(target_y, predicted_y):
        """Compute the macro soft F1-score as a cost.
        Average (1 - soft-F1) across all labels.
        Use probability values instead of binary predictions.
        
        Args:
            y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
            y_hat (float32 Tensor): probability matrix of shape (BATCH_SIZE, N_LABELS)
            
        Returns:
            cost (scalar Tensor): value of the cost function for the batch
        """
        
        target_y = tf.cast(target_y, tf.float32)
        predicted_y = tf.cast(predicted_y, tf.float32)
        tp = tf.reduce_sum(predicted_y * target_y, axis=0)
        fp = tf.reduce_sum(predicted_y * (1 - target_y), axis=0)
        fn = tf.reduce_sum((1 - predicted_y) * target_y, axis=0)
        soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
        cost = 1 - soft_f1 # reduce 1 - soft-f1 in order to increase soft-f1
        macro_cost = tf.reduce_mean(cost) # average on all labels
        
        return macro_cost

    @staticmethod
    def macro_f1(target_y, predicted_y, thresh=0.5):
        """Compute the macro F1-score on a batch of observations (average F1 across labels)
        
        Args:
            y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
            y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
            thresh: probability value above which we predict positive
            
        Returns:
            macro_f1 (scalar Tensor): value of macro F1 for the batch
        """
        y_pred = tf.cast(tf.greater(predicted_y, thresh), tf.float32)
        tp = tf.cast(tf.math.count_nonzero(y_pred * target_y, axis=0), tf.float32)
        fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - target_y), axis=0), tf.float32)
        fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * target_y, axis=0), tf.float32)
        f1 = 2*tp / (2*tp + fn + fp + 1e-16)
        macro_f1 = tf.reduce_mean(f1)
        return macro_f1

    @staticmethod
    def macro_double_soft_f1(y, y_hat):
        """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
        Use probability values instead of binary predictions.
        This version uses the computation of soft-F1 for both positive and negative class for each label.
        
        Args:
            y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
            y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
            
        Returns:
            cost (scalar Tensor): value of the cost function for the batch
        """
        y = tf.cast(y, tf.float32)
        y_hat = tf.cast(y_hat, tf.float32)
        tp = tf.reduce_sum(y_hat * y, axis=0)
        fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
        fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
        tn = tf.reduce_sum((1 - y_hat) * (1 - y), axis=0)
        soft_f1_class1 = 2*tp / (2*tp + fn + fp + 1e-16)
        soft_f1_class0 = 2*tn / (2*tn + fn + fp + 1e-16)
        cost_class1 = 1 - soft_f1_class1 # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
        cost_class0 = 1 - soft_f1_class0 # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
        cost = 0.5 * (cost_class1 + cost_class0) # take into account both class 1 and class 0
        macro_cost = tf.reduce_mean(cost) # average on all labels
        return macro_cost

    def create_model(self, pre_trained_model):
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
            prediction_layer_one = tf.keras.layers.Dense(4096)
            prediction_layer_two = tf.keras.layers.Dense(39,
                                                         activation='sigmoid')

            self.model = tf.keras.Sequential([
                base_model, global_average_layer, prediction_layer_one,
                prediction_layer_two
            ])

            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss= [lambda y_true,y_pred: Model.macro_double_soft_f1(y_true, y_pred)],
                metrics=[tf.keras.metrics.AUC(num_thresholds=10000, multi_label=True), 
                                            tf.keras.metrics.AUC(curve='PR', num_thresholds=10000, multi_label=True), 
                                            Model.macro_f1,
                                            tf.keras.metrics.Recall(),
                                            tf.keras.metrics.Precision()])
            
        if Config.MODE == 'predict_images' and pre_trained_model is not None:
            self.model.load_weights(pre_trained_model)

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
                                                               min_lr=1e-6)
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

    def predict(self, images):
        predictions = self.model.predict(images)
        decoded_predictions = list()

        for prediction in predictions:
            decoded_prediction = dict()

            decoded_prediction['gender'] = self.ds.one_hot_decoding[np.argmax(prediction[0:2])]
            decoded_prediction['articleType'] = self.ds.one_hot_decoding[np.argmax(prediction[2:17]) + 2]
            decoded_prediction['baseColour'] = self.ds.one_hot_decoding[np.argmax(prediction[17:28]) + 17]
            decoded_prediction['season'] = self.ds.one_hot_decoding[np.argmax(prediction[28:32]) + 28]
            decoded_prediction['usage'] = self.ds.one_hot_decoding[np.argmax(prediction[32:]) + 32]

            decoded_predictions.append(decoded_prediction)
        
        return decoded_predictions
