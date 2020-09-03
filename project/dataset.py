import sys
from pathlib import Path

import pandas as pd
import tensorflow as tf

from config import Config
from utils.data_utils import DataUtils

AUTOTUNE = tf.data.experimental.AUTOTUNE


class DataSet(object):
    def __init__(self):
        self.one_hot_decoding = {
            0: 'Men',
            1: 'Women',
            2: 'Bra',
            3: 'Briefs',
            4: 'Dresses',
            5: 'Jackets',
            6: 'Jumpsuit',
            7: 'Kurtas',
            8: 'Pants',
            9: 'Shirts',
            10: 'Shoes',
            11: 'Shorts',
            12: 'Skirts',
            13: 'Socks',
            14: 'Sweater',
            15: 'Tops',
            16: 'Tshirts',
            17: 'Black',
            18: 'Blue',
            19: 'Brown',
            20: 'Green',
            21: 'Grey',
            22: 'Orange',
            23: 'Pink',
            24: 'Purple',
            25: 'Red',
            26: 'White',
            27: 'Yellow',
            28: 'Fall',
            29: 'Spring',
            30: 'Summer',
            31: 'Winter',
            32: 'Casual',
            33: 'Ethnic',
            34: 'Formal',
            35: 'Party',
            36: 'Smart Casual',
            37: 'Sports',
            38: 'Travel'
        }

    def _get_label(self, id):
        # Maps id to one hot encoded tensor. Acts on one sample at a time.
        # Dimensions of output tensor [2, 12, 11, 4, 7]
        hotencoded_tensors = [
            tf.one_hot(self.df.loc[id.numpy()][key], len(value))
            for key, value in self.categories.items()
        ]
        return tf.concat(hotencoded_tensors, axis=0)

    def _process_id(self, id):
        label = tf.py_function(func=self._get_label, inp=[id], Tout=tf.float32)
        file_path = tf.strings.join([
            str(Config.DATA_DIR), '/images/',
            tf.strings.as_string(id), '.jpg'
        ])
        # Load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = DataUtils.decode_img(img)

        # Set shape manually bc tensor is returned by a py_func
        label.set_shape([39])
        return img, label

    def _prepare_for_training(self,
                              split_name,
                              split,
                              shuffle_buffer_size=1000):
        # Use a cache dir for the normal sized dataset
        split = split.cache(str(Path(self.data_dir, 'cache')))
        split = split.shuffle(buffer_size=shuffle_buffer_size)

        # Repeat forever
        split = split.repeat()
        split = split.batch(Config.BATCH_SIZE)
        # `prefetch` lets the dataset fetch batches in the background while the
        # model is training.
        return split.prefetch(buffer_size=AUTOTUNE)

    def create(self, splits=Config.DS_SPLIT, data_dir=Config.DATA_DIR):
        # ds sizes:
        # test      4445
        # train    44199
        # val       4444
        self.data_dir = data_dir
        # Saves the categories of the labels
        self.categories = dict()
        # Does the data directory exist?
        if not self.data_dir.exists():
            sys.exit(
                'No dataset for training found at the given data directory')
        Config.STYLES_USE_COLS.append('split')
        df = DataUtils.load_data_frame('adjusted_styles.csv')
        set_names = ['train', 'val', 'test']
        ids_by_split = [list(df[df['split'] == x].index) for x in set_names]
        df = df.drop('split', axis=1)
        for col in list(df.columns):
            df[col] = pd.Categorical(df[col])
            self.categories[col] = df[col].cat.categories
            df[col] = df[col].cat.codes

        self.df = df

        for set_name, indices in list(zip(set_names, ids_by_split)):
            ds = tf.data.Dataset.from_tensor_slices(tf.constant(indices))
            ds = ds.map(self._process_id, num_parallel_calls=AUTOTUNE)
            ds = self._prepare_for_training(set_name, ds)
            setattr(self, set_name, ds)
        return self
