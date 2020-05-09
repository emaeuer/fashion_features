import time
import tensorflow as tf

from utils import DataSet
from config import Config
from evaluation import Evaluation

default_timeit_steps = 1000


def timeit(ds, steps=default_timeit_steps):
    # tests the performance of loading and preprocessing the dataset
    start = time.time()
    it = iter(ds)
    for i in range(steps):
        next(it)
        if i % 10 == 0:
            print(i, end='')
    print()
    end = time.time()

    duration = end - start
    print("{} batches: {} s".format(steps, duration))
    print("{:0.5f} Images/s".format(Config.BATCH_SIZE * steps / duration))


if __name__ == '__main__':
    ds = DataSet()
    # timeit(ds.train)

    # Evaluation()
    # DataSet.adjust_styles_csv()
