import numpy as np
from scipy.spatial import ConvexHull
import laspy
from data_visualise import Visibility

def main():

    visibility = Visibility(area_names=[0, 1, 2, 3], sampling="random", sample_ratio=100)

    visibility.import_data()

    p_array = np.array([[-18122.7802, -59875.4444, 10]])

    visibility.raycast(p_array, n_xy=500000, n_xz=10000)

    visibility.colour(colour='original')

    visibility.export_data(version='1-1')

    # Get (x, y) coordinates of red points
    visible_geo_array = visibility.points_array[np.where(visibility.visible_count_array > 0)[0] :2]

    # Calculate furthest Manhattan distance
    distance_array = np.zeros((visibility.n_p, len(visible_geo_array)))
    for i in range(visibility.n_p):
        distance_array[i, :] = np.sum(np.abs(visible_geo_array - p_array[i, :2]), axis=1)
    max_distance = np.max(distance_array.ravel())

    visibility = Visibility(area_names=[0, 1, 2, 3], sampling="specific", max_distance=max_distance)

    visibility.import_data()

    p = np.array([[-18122.7802, -59875.4444, 10]])

    visibility.raycast(p, n_xy=500000, n_xz=10000)

    visibility.colour(colour='original')

    visibility.export_data(version='2-1')


if __name__ == "__main__":
    main()