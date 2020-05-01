import sys

import yaml
import tensorflow as tf
from pathlib import Path
import pandas as pd
import numpy as np

from config import Config

AUTOTUNE = tf.data.experimental.AUTOTUNE


class DataSet(object):
    def __init__(self, data_dir=Config.DATA_DIR):
        self.data_dir = data_dir
        # Saves the categories of the labels
        self.categories = dict()
        # Does the data directory exist?
        if not self.data_dir.exists():
            sys.exit(
                'No dataset for training found at the given data directory')
        self._read_labels()

    @staticmethod
    def load_styles_data_frame():
        df = pd.read_csv(Path(Config.DATA_DIR, 'styles.csv'),
                         usecols=Config.STYLES_USE_COLS)
        df.set_index(Config.STYLES_INDEX_COL, inplace=True)
        df.drop(Config.IDS_TO_SKIP, inplace=True)
        return df

    @staticmethod
    def remap_labels(df: pd.DataFrame,
                     mappings_file_path=Config.MAPPINGS_FILE):
        # Remaps labels according to a predefined mapping
        mappings = yaml.full_load(open(mappings_file_path, 'r'))
        for key, value in mappings.items():
            df[key] = df[key].map(value)
        return df.dropna()

    def _read_labels(self):
        # Returns labels as a endcode dataframe. Labels are indexed by their
        # ids
        df = DataSet.load_styles_data_frame()
        df = DataSet.remap_labels(df)
        for col in df.columns:
            df[col] = pd.Categorical(df[col])
            self.categories[col] = df[col].cat.categories
            df[col] = df[col].cat.codes
        self.df = df

    def _get_label(self, id):
        # Maps id to one hot encoded tensor. Acts on one sample at a time.
        # Dimensions of output tensor [2, 12, 11, 4, 7]
        hotencoded_tensors = [
            tf.one_hot(self.df.loc[id.numpy()][key], value.size)
            for key, value in self.categories.items()
        ]
        return tf.concat(hotencoded_tensors, axis=0)

    def _decode_img(self, img):
        # Convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # Resize the image to the desired size.
        return tf.image.resize(img, [Config.IMG_WIDTH, Config.IMG_HEIGHT])

    def _process_id(self, id):
        label = tf.py_function(func=self._get_label, inp=[id], Tout=tf.float32)
        file_path = tf.strings.join([
            str(Config.DATA_DIR), '/images/',
            tf.strings.as_string(id), '.jpg'
        ])
        # Load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = self._decode_img(img)
        return img, label

    def _augment(self, image, label):
        # TODO add more augmentation
        return tf.image.random_crop(image, size=[28, 28, 1]), label

    def _prepare_for_training(self,
                              split_name,
                              split,
                              shuffle_buffer_size=1000):
        # Use a cache dir for the normal sized dataset
        split = split.cache(str(Path(self.data_dir, 'cache')))
        split = split.shuffle(buffer_size=shuffle_buffer_size)

        # augment the training data
        if split_name == 'train':
            split = split.map(self._augment, num_parallel_calls=AUTOTUNE)

        # Repeat forever
        split = split.repeat()
        split = split.batch(Config.BATCH_SIZE)
        # `prefetch` lets the dataset fetch batches in the background while the
        # model is training.
        return split.prefetch(buffer_size=AUTOTUNE)

    def create(self, splits=Config.DS_SPLIT):
        ids = self.df.index.to_numpy()
        np.random.shuffle(ids)
        # Converts splits from percentages to absolute values
        splits = list(map(lambda x: int(x * len(ids)), splits))
        splitted_ids = np.split(ids, splits)

        for split_name, split_of_ids in zip(['train', 'validate', 'test'],
                                            splitted_ids):
            split = tf.data.Dataset.from_tensor_slices(
                tf.constant(split_of_ids))
            split = split.map(self._process_id, num_parallel_calls=AUTOTUNE)
            split = self._prepare_for_training(split_name, split)
            setattr(self, split_name, split)

        return self
