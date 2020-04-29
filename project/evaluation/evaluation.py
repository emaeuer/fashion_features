from pathlib import Path
from collections import Counter

import yaml
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from utils import DataSet
from config import Config


class Evaluation(object):
    def __init__(self):
        self.df = DataSet.load_styles_data_frame()
        self.counts = dict()
        self.results_path = Path(Config.VIZ_RESULTS_DIR)
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.evaluate_data()

    def evaluate_data(self):
        """Create a plot for the column of a data frame"""
        for col in self.df:
            self.counts[col] = self.df[col].value_counts()
            self._plot_dist(col)

        yaml.dump({k: v.to_dict()
                   for k, v in self.counts.items()},
                  Path(self.results_path, 'counts.yaml').open('w'))

    def _plot_dist(self, col):
        self.counts[col].plot.bar(y='count', rot=0, figsize=(10, 10))
        plt.savefig(Path(self.results_path, f'{col}.png'))
        plt.close()

    def count_img_sizes(self):
        sizes = list()
        for id in tqdm(self.df.index):
            img_path = Path(Config.DATA_DIR, 'images', f'{id}.jpg')
            if not img_path.exists():
                print(f'Error: ID {id} has no corresponding image!')
            else:
                sizes.append(Image.open(img_path).size)
        print(Counter(sizes))
