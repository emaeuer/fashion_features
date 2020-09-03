import argparse
import os
from utils.data_utils import DataUtils
from dataset import DataSet
from model import Model
from evaluation import Evaluation
from config import Config
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create and evaluate a multi-class \
classification task on the fashinon feature dataset")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', dest='train', action='store_true')
    group.add_argument('--evaluate', dest='eval', action='store_true')
    group.add_argument('--predict_images',
                       type=str,
                       dest='predict_images',
                       help="Path to the test images",
                       default=None)
    group.add_argument('--adjust_data',
                       dest='adjust_data',
                       action='store_true',
                       help='Preprocess Data (augmentation not included)')
    group.add_argument('--analyze_data',
                       dest='analyze_data',
                       action='store_true',
                       help='Plot label distributions')

    parser.add_argument('--load_model',
                        type=str,
                        dest='load_model',
                        help="Path of the model weights to load",
                        default=None)

    args = parser.parse_args()
    if args.train:
        if not Config.LOG_DIR.exists():
            Config.LOG_DIR.mkdir(parents=True)
        if not Config.CHECKPOINT_DIR.exists():
            Config.CHECKPOINT_DIR.mkdir(parents=True)
        model = Model(DataSet().create())
        model.fit()
    if args.eval:
        model = Model(DataSet().create(), args.load_model)
        history = model.eval()
        print('hello')
    if args.adjust_data:
        DataUtils.adjust_data()
    if args.analyze_data:
        if not Config.VIZ_RESULTS_DIR.exists():
            Config.VIZ_RESULTS_DIR.mkdir()
        Evaluation()
    if args.predict_images is not None:
        model = Model(DataSet(), args.load_model)
        images = DataUtils.load_all_images(args.predict_images)
        predictions = model.predict(images)

        for img, pred in zip(images, predictions):
            plt.figure(figsize=(20, 20))
            plt.imshow(img)
            plt.title(pred)
            plt.show()
