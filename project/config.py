import os
import datetime
from pathlib import Path
from dotenv import load_dotenv

from utils.gpu_utils import gpu_memory_map
basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))


class Config(object):
    IMG_SHAPE = (int(os.environ.get('IMG_HEIGHT')),
                 int(os.environ.get('IMG_WIDTH')))
    BATCH_SIZE = int(os.environ.get('BATCH_SIZE'))
    EPOCHS = int(os.environ.get('EPOCHS'))
    DS_SPLIT = [0.8, 0.1, 0.1]

    STYLES_USE_COLS = [
        'id', 'gender', 'articleType', 'baseColour', 'season', 'usage'
    ]
    STYLES_INDEX_COL = 'id'
    MIN_SAMPLES_PER_LABEL = int(os.environ.get('MIN_SAMPLES_PER_LABEL'))

    MAPPINGS_FILE = Path(os.environ.get('MAPPINGS_FILE'))
    VIZ_RESULTS_DIR = Path(os.environ.get('VIZ_RESULTS_DIR'))
    DATA_DIR = Path(basedir, os.environ.get('DATA_DIR'))
    LOG_DIR = Path(
        f'{os.environ["LOG_DIR"]}/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    )
    CHECKPOINT_DIR = Path(
        f'{os.environ["CHECKPOINT_DIR"]}/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    )

    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
            [str(x) for x, y in gpu_memory_map().items() if y == 0])
