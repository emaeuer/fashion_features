from utils.data_set import DataSet
import time

default_timeit_steps = 1000


def timeit(ds, steps=default_timeit_steps):
    # tests the performance of loading and preprocessing the dataset
    start = time.time()
    it = iter(ds)
    for i in range(steps):
        batch = next(it)
        if i % 10 == 0:
            print(i, end='')
    print()
    end = time.time()

    duration = end - start
    print("{} batches: {} s".format(steps, duration))
    print("{:0.5f} Images/s".format(BATCH_SIZE * steps / duration))


if __name__ == '__main__':
    ds = DataSet(small=True).create()
    timeit(ds)
