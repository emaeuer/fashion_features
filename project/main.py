import argparse
import os
from utils.data_utils import DataUtils
from dataset import DataSet
from model import Model
from evaluation import Evaluation
from config import Config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--adjust_data',
                        dest='adjust_data',
                        action='store_true')
    parser.add_argument('--analyze_data',
                        dest='analyze_data',
                        action='store_true')

    args = parser.parse_args()
    if args.train:
        Config.MODE = 'train'
        if not Config.LOG_DIR.exists():
            Config.LOG_DIR.mkdir(parents=True)
        if not Config.CHECKPOINT_DIR.exists():
            Config.CHECKPOINT_DIR.mkdir(parents=True)
        model = Model(DataSet())
        model.fit()
        model.eval()
    if args.adjust_data:
        Config.MODE = 'adjust_data'
        DataUtils.adjust_data()
    if args.analyze_data:
        Config.MODE = 'analyze_data'
        if not Config.VIZ_RESULTS_DIR.exists():
            Config.VIZ_RESULTS_DIR.mkdir()
        Evaluation()
