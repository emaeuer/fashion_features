import os
from pathlib import Path
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))


class Config(object):
    DATA_DIR = Path(basedir,
                    os.environ.get('DATA_DIR') or 'data/fashion-dataset')
    MAPPINGS_FILE = os.environ.get('MAPPINGS_FILE') or 'data/mappings.yaml'
    VIZ_RESULTS_DIR = os.environ.get('VIZ_RESULTS_DIR') or 'results'
    IMG_SHAPE = (os.environ.get('IMG_HEIGHT')
                 or 331, os.environ.get('IMG_WIDTH') or 331)
    BATCH_SIZE = os.environ.get('BATCH_SIZE') or 32
    STYLES_INDEX_COL = 'id'
    STYLES_USE_COLS = [
        'id', 'gender', 'articleType', 'baseColour', 'season', 'usage'
    ]
    DS_SPLIT = [0.8, 0.1, 0.1]
    MIN_SAMPLES_PER_LABEL = 20
