import os
from pathlib import Path
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))


class Config(object):
    DATA_URI = os.environ.get('DATA_URI')
    DATA_URI_SMALL = os.environ.get('DATA_URI_SMALL')
    DATA_DIR = Path(basedir, os.environ.get('DATA_DIR') or 'data')
    IMG_WIDTH = os.environ.get('IMG_Width') or 256
    IMG_HEIGHT = os.environ.get('IMG_HEIGHT') or 256
    BATCH_SIZE = os.environ.get('BATCH_SIZE') or 32
