import itertools
import multiprocessing
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
        self._create()

    @staticmethod
    def adjust_styles_csv(mappings_file_path=Config.MAPPINGS_FILE):
        np.random.seed(42)
        df = DataSet.load_data_frame()
        new_start_id = max(df.index) + 1
        mappings = yaml.full_load(open(mappings_file_path, 'r'))
        for key, value in mappings.items():
            df[key] = df[key].map(value)
        df.dropna()
        # Converts three relative splits to the two index break points
        splits = np.cumsum([int(x * len(df)) for x in Config.DS_SPLIT])[:-1]
        # Randomizes data and creates splits
        train, val, test = np.split(df.sample(frac=1, random_state=42), splits)
        grouped = train.groupby(list(
            df.columns)).apply(lambda x: list(x.index) if len(x) < Config.
                               MIN_SAMPLES_PER_LABEL else None).dropna()
        old_ids = list()
        duplicated_labels = list()
        for k, v in grouped.items():
            nrb_of_items_to_fill_up = Config.MIN_SAMPLES_PER_LABEL - len(v)
            chosen_old_ids = np.random.choice(v, nrb_of_items_to_fill_up)
            old_ids.extend(chosen_old_ids)
            duplicated_labels.extend([k] * nrb_of_items_to_fill_up)

        train_extension = pd.DataFrame(duplicated_labels, columns=df.columns)
        # shift index to get new unique ids
        train_extension.index += new_start_id
        # Create adjusted styles dataframe
        df = df.append(train_extension)
        df.loc[val.index, 'split'] = 'val'
        df.loc[test.index, 'split'] = 'test'
        df.loc[train.index.union(train_extension.index), 'split'] = 'train'
        df['id'] = df.index
        df.to_csv(Path(Config.DATA_DIR, 'adjusted_styles.csv'))

        pool = multiprocessing.Pool()
        progress_bar = tqdm(total=len(old_ids))

        def update(*a):
            progress_bar.update()

        for i in list(zip(old_ids, train_extension.index)):
            pool.apply_async(DataSet.add_augmented_images,
                             args=i,
                             callback=update)

        pool.close()
        pool.join()

        print('Finished with dataset adjustment')

    @staticmethod
    def add_augmented_images(*ids):
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        # Takes a list of ids containing the old id and the new id
        # Applies augmentation to the old image and saves it as a new one
        image_paths = [
            str(Path(Config.DATA_DIR, 'images', f'{id}.jpg')) for id in ids
        ]
        image = tf.image.decode_jpeg(tf.io.read_file(image_paths[0]))
        image = tf.image.resize_with_pad(image, Config.IMG_HEIGHT,
                                         Config.IMG_WIDTH)
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
    def load_data_frame(file_name='styles.csv'):
        df = pd.read_csv(Path(Config.DATA_DIR, file_name),
                         usecols=Config.STYLES_USE_COLS)
        df.set_index(Config.STYLES_INDEX_COL, inplace=True)
        ids_to_skip = set(Config.IDS_TO_SKIP).intersection(set(df.index))
        if len(ids_to_skip) > 0:
            df.drop(ids_to_skip, inplace=True)
        return df

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
        return tf.image.resize_with_pad(img, Config.IMG_HEIGHT,
                                        Config.IMG_WIDTH)

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

    def _create(self, splits=Config.DS_SPLIT):
        Config.STYLES_USE_COLS.append('split')
        df = DataSet.load_data_frame('adjusted_styles.csv')
        set_names = ['train', 'val', 'test']
        ids_by_split = [list(df[df['split'] == x].index) for x in set_names]
        df = df.drop('split', axis=1)
        for col in list(df.columns):
            df[col] = pd.Categorical(df[col])
            self.categories[col] = df[col].cat.categories
            df[col] = df[col].cat.codes
        all_labels = list(itertools.chain(*self.categories.values()))
        self.categories = dict(zip(range(len(all_labels)), all_labels))
        self.df = df
        for set_name, indices in list(zip(set_names, ids_by_split)):
            ds = tf.data.Dataset.from_tensor_slices(tf.constant(indices))
            ds = ds.map(self._process_id, num_parallel_calls=AUTOTUNE)
            ds = self._prepare_for_training(set_name, ds)
            setattr(self, set_name, ds)
        return ids_by_split
