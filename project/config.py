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
    IMG_WIDTH = os.environ.get('IMG_Width') or 256
    IMG_HEIGHT = os.environ.get('IMG_HEIGHT') or 256
    BATCH_SIZE = os.environ.get('BATCH_SIZE') or 32
    STYLES_INDEX_COL = 'id'
    STYLES_USE_COLS = [
        'id', 'gender', 'articleType', 'baseColour', 'season', 'usage'
    ]
    IDS_TO_SKIP = [39403, 39410, 39401, 39425]
