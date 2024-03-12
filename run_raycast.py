import laspy
from data_visualise import Visibility
import numpy as np

def main():

    visibility = Visibility(area_names=[0, 1, 2, 3], sampling="random", sample_ratio=100)

    visibility.import_data()

    p = np.array([[-18122.7802, -59875.4444, 10]])

    visibility.raycast(p, n_xy=500000, n_xz=10000)

    visibility.colour(colour='original')

    visibility.export_data(version='2-1')

if __name__ == "__main__":
    main()
