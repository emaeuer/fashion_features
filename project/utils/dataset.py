import multiprocessing
import random
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

import yaml
from tqdm import tqdm

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
    def adjust_styles_csv(mappings_file_path=Config.MAPPINGS_FILE):
        df = DataSet.load_styles_data_frame()
        new_start_id = max(df.index) + 1
        mappings = yaml.full_load(open(mappings_file_path, 'r'))
        for key, value in mappings.items():
            df[key] = df[key].map(value)
        df.dropna()
        grouped = df.groupby(list(df.columns))
        old_ids = list()
        duplicated_labels = list()
        h = 0
        print(grouped.count())
        # for k, v in grouped.indices.items():
        #     test = set(v) - set(df.index)
        #     if len(test) > 0:
        #         print(test)
        #     if len(v) >= Config.MIN_SAMPLES_PER_LABEL:
        #         continue
        #     for i in range(Config.MIN_SAMPLES_PER_LABEL - len(v)):
        #         chosen_old_id = random.choice(v)
        #         old_ids.append(chosen_old_id)
        #         duplicated_labels.append(k)
        # print(h)
        # df_extension = pd.DataFrame(duplicated_labels, columns=df.columns)
        # # shift index to get new unique ids
        # df_extension.index += new_start_id
        # df = df.append(df_extension)
        # df['id'] = df.index
        # df.to_csv(Path(Config.DATA_DIR, 'adjusted_styles.csv'))
        # pool = multiprocessing.Pool(1)
        # progress_bar = tqdm(total=len(old_ids))

        # def update(*a):
        #     progress_bar.update()

        # for i in [list(zip(old_ids, df_extension.index))[0]]:
        #     pool.apply_async(DataSet.add_augmented_images,
        #                      args=i,
        #                      callback=update)

        # pool.close()
        # pool.join()

        # print('Finished with dataset adjustment')

    @staticmethod
    def add_augmented_images(*ids):
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        # Takes a list of ids containing the old id and the new id
        # Applies augmentation to the old image and saves it as a new one
        image_paths = [
            str(Path(Config.DATA_DIR, 'images', f'{id}.jpg')) for id in ids
        ]
        image = tf.image.decode_jpeg(tf.io.read_file(image_paths[0]))
        dg = ImageDataGenerator(
            rotation_range=30,
            zoom_range=0.5,
            shear_range=0.3,
            horizontal_flip=True,
            width_shift_range=0.3,
            height_shift_range=0.3,
        )

        tf.keras.preprocessing.image.save_img(
            image_paths[1], dg.random_transform(image.numpy()))

    @staticmethod
    def load_styles_data_frame(file_name='styles.csv'):
        df = pd.read_csv(Path(Config.DATA_DIR, file_name),
                         usecols=Config.STYLES_USE_COLS)
        df.set_index(Config.STYLES_INDEX_COL, inplace=True)
        if 'adjusted' not in file_name:
            df.drop(Config.IDS_TO_SKIP, inplace=True)
        return df

    def _read_labels(self):
        # Returns labels as a endcode dataframe. Labels are indexed by their
        # ids
        df = DataSet.load_styles_data_frame()
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
        # img = tf.image.convert_image_dtype(img, tf.float32)
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
        # return image, label

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
