import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
import tensorflow as tf

from config import Config


class DataUtils:
    @staticmethod
    def adjust_data(mappings_file_path=Config.MAPPINGS_FILE):
        np.random.seed(42)
        df = DataUtils.load_data_frame()
        df = DataUtils.drop_missing_images(df)
        new_start_id = max(df.index) + 1
        mappings = yaml.full_load(open(mappings_file_path, 'r'))
        for key, value in mappings.items():
            df[key] = df[key].map(value)
        df.dropna(inplace=True)
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

        print('Adding augmented images...')
        for i in list(zip(old_ids, train_extension.index)):
            pool.apply_async(DataUtils.add_augmented_images,
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
        image = tf.image.resize(image, Config.IMG_SHAPE)
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
        return df

    @staticmethod
    def drop_missing_images(df):
        image_path_ids = set(
            int(path.stem)
            for path in Path(Config.DATA_DIR, 'images').glob('*'))
        return df.drop(index=set(df.index) - image_path_ids)