import csv
import os

from config import Config

import tensorflow as tf
import numpy as np
from pathlib import Path

AUTOTUNE = tf.data.experimental.AUTOTUNE


class DataSet(object):
    def __init__(self, small=False, data_dir=Config.DATA_DIR):
        self.small = small
        self.data_dir = data_dir
        self.fname = 'fashion-product-images-' \
            f'{"small" if self.small else "dataset"}'
        self.data_subdir = Path(self.data_dir, self.fname)
        # download data if no extracted data-folder exists
        if not self.data_subdir.exists():
            origin = Config.DATA_URI_SMALL if self.small else Config.DATA_URI
            tf.keras.utils.get_file(origin=origin,
                                    fname=self.fname,
                                    cache_dir=self.data_dir,
                                    cache_subdir=self.data_subdir,
                                    extract=True)

    def _get_id_label_mapping(self):
        reader = csv.DictReader(open(str(Path(self.data_subdir,
                                              'styles.csv'))))
        return {
            rows['id']: np.array(list(rows.values())[1:])
            for rows in reader
        }

    def _get_label(self, file_path):
        file_path_parts = tf.strings.split(file_path, os.path.sep)
        file_name = file_path_parts[-1]
        id = Path(bytes.decode(file_name.numpy())).stem
        return tf.convert_to_tensor(self.id_to_labels[id])

    def _decode_img(self, img):
        # Convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # Resize the image to the desired size.
        return tf.image.resize(img, [Config.IMG_WIDTH, Config.IMG_HEIGHT])

    def _process_path(self, file_path):
        label = tf.py_function(
            func=self._get_label,
            inp=[file_path],
            Tout=tf.string
        )
        # Load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = self._decode_img(img)
        return img, label

    def _prepare_for_training(self, shuffle_buffer_size=1000):
        # Use a cache dir for the normal sized dataset
        if not self.small:
            self.ds = self.cache(Path(self.data_subdir, 'cache'))
        else:
            self.ds = self.ds.cache()
        self.ds = self.ds.shuffle(buffer_size=shuffle_buffer_size)
        # Repeat forever
        self.ds = self.ds.repeat()
        self.ds = self.ds.batch(BATCH_SIZE)
        # `prefetch` lets the dataset fetch batches in the background while the
        # model is training.
        self.ds = self.ds.prefetch(buffer_size=AUTOTUNE)

    def create(self):
        self.ds = tf.data.Dataset.list_files(
            str(f'{self.data_subdir}/images/*'))
        self.id_to_labels = self._get_id_label_mapping()
        self.ds = self.ds.map(self._process_path, num_parallel_calls=AUTOTUNE)
        self._prepare_for_training()
        return self.ds
