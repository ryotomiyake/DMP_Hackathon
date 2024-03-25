import numpy as np
import laspy
import tqdm
import os
import re
import joblib

class Visibility:
    def __init__(self, area_names):
        # Initialize Visibility object

        self.area_names = area_names if isinstance(area_names, list) else [area_names]
        self.public_header_list = []
        self.n_points_total = 0
        self.points_array = np.empty((0, 6), dtype=float)
        self.p_array = None
        self.n_p = 0
        self.visible_indices_array = np.array([], dtype=int)
        self.visible_count_array = None
        self.include_p = False
        self.no_points_p = 0

    def generate_file_path(self, area_name):

        file_name = [r"Area0_230713_031852_S0", r"Area1_230724_065805_S0", r"Area2_230724_054607_S0", r"Area3_230724_011027_S0"][area_name]

        if self.sampling == "random":
            file_path = file_name + "_random_1_" + str(self.sample_ratio) + ".laz"
        else:
            file_path = file_name + ".laz"

        return file_path
    

    def import_split_data(self, file_path, sample_ratio, sampling_centre, sampling_distance):

        with laspy.open(file_path, mode='r') as las:
            n_points = las.header.point_count
            points = las.points
            indices = np.random.choice(len(points), size=int(n_points/sample_ratio), replace=False)
            file_points_array = np.column_stack((las.points['x'], las.points['y'], las.points['z'], las.points['red'], las.points['green'], las.points['blue']))[indices]

            # Filter points within sampling_distance from sampling_centre
            distance_array = np.sum(np.abs(file_points_array[:, :2] - sampling_centre), axis=1)
            in_range_array = distance_array <= sampling_distance
            file_points_array = file_points_array[in_range_array]

        return file_points_array


    def import_data(self, chunk_size=1000_000, sampling="random", sample_ratio=1000, sampling_centre=np.array((0, 0, 0)), sampling_distance=1000):
        # Import data from LAS file

        # Store sampling method
        self.sampling = sampling
        self.sample_ratio = sample_ratio

        if self.sampling == "random":

            for area_name in self.area_names:

                file_path = self.generate_file_path(area_name)

                with laspy.open(file_path, mode='r') as las:
                    self.public_header_list.append(las.header)
                    n_points_added = las.header.point_count

                    file_points_array = np.full((n_points_added, 6), np.nan)
                    current_index = 0
                    for points in tqdm.tqdm(las.chunk_iterator(chunk_size)):
                        file_points_array[current_index:current_index+chunk_size, 0] = points['x']
                        file_points_array[current_index:current_index+chunk_size, 1] = points['y']
                        file_points_array[current_index:current_index+chunk_size, 2] = points['z']
                        file_points_array[current_index:current_index+chunk_size, 3] = points['red']
                        file_points_array[current_index:current_index+chunk_size, 4] = points['green']
                        file_points_array[current_index:current_index+chunk_size, 5] = points['blue']
                        current_index += chunk_size

                self.points_array = np.vstack((self.points_array, file_points_array))
                self.n_points_total += n_points_added

        elif self.sampling == "specific":

            file_paths = []
            for area_name in self.area_names:
                pattern = re.compile(r"Area" + str(area_name) + r"_[0-9A-Z_]+_split_\d+\.laz")
                directory = "split_laz"
                for file_name in os.listdir(directory):
                    if pattern.match(file_name):
                        file_paths.append(os.path.join(directory, file_name))

            results = joblib.Parallel(n_josb=-1)(
                joblib.delayed(self.import_split_data)(file_path, sample_ratio, sampling_centre, sampling_distance) for file_path in file_paths
            )

            # Update points_array and n_points_total
            self.points_array = np.vstack(results)
            self.n_points_total = len(self.points_array)


    def raycast(self, p_array, n_xy, n_xz):

        if len(p_array.shape) == 2 and p_array.shape[1] == 3:
            pass
        else:
            raise ValueError("Invalid shape for p_array")
        
        self.p_array = p_array
        self.n_p = self.p_array.shape[0]

        for i, single_p in enumerate(self.p_array):
            print("** {}/{} **".format(i + 1, self.n_p))
            self._raycast_single_p(single_p, n_xy, n_xz)

        # 
        self.visible_count_array = np.bincount(self.visible_indices_array)


    def _raycast_single_p(self, single_p, n_xy, n_xz):
        # Format points_array for visibility calculations
        # Perform ray casting to determine visibility

        # Create an array to store data
        points_geo_array = np.full(self.points_array.shape, np.nan)
        points_geo_array[:, :3] = self.points_array[:, :3]
        
        # Translate points relative to observed
        print("Step 1/7")
        points_geo_array[:, :3] = points_geo_array[:, :3] - single_p

        # Calculate azimuth angle (xy-plane), divide into n_xy increments and replace value with number of increments
        print("Step 2/7")
        points_geo_array[:, 3] = np.arctan2(points_geo_array[:, 1], points_geo_array[:, 0])
        xy_min = np.min(points_geo_array[:, 3])
        xy_max = np.max(points_geo_array[:, 3])
        h_xy = (xy_max - xy_min) / n_xy
        points_geo_array[:, 3] = ((points_geo_array[:, 3] - xy_min) // h_xy).astype(int)

        # Calculate elevation angle (xz-plane), divide into n_xy increments and replace value with number of increments
        print("Step 3/7")
        points_geo_array[:, 4] = np.arctan2(points_geo_array[:, 2], points_geo_array[:, 0])
        xz_min = np.min(points_geo_array[:, 3])
        xz_max = np.max(points_geo_array[:, 3])
        h_xz = (xz_max - xz_min) / n_xz
        points_geo_array[:, 4] = ((points_geo_array[:, 4] - xz_min) // h_xz).astype(int)

        # Calculate distance from observer
        print("Step 4/7")
        points_geo_array[:, 5] = np.linalg.norm(points_geo_array[:, :3], axis=1)
        points_geo_array[:, 5][points_geo_array[:, 5] == 0] = np.inf  # Set distance to observer as infinite

        # Sort points_geo_array for efficient ray casting
        print("Step 5/7")
        order_array = np.lexsort((points_geo_array[:, 5], points_geo_array[:, 4], points_geo_array[:, 3]))
        points_geo_array = points_geo_array[order_array]
        
        #  Find unique pairs of azimuth and elevation angle
        print("Step 6/7")
        unique_pairs, inverse_indices = np.unique(points_geo_array[:, 3:5], axis=0, return_inverse=True)

        # Store index of closest point (visible point) for each pair of azimuth and elevation angle
        print("Step 7/7")
        visible_indices_array_signle_p = np.full((len(unique_pairs)), self.n_points_total + 1, dtype=int)
        for idx, dir_group in enumerate(tqdm.tqdm(inverse_indices)):
            if idx < visible_indices_array_signle_p[dir_group]:
                visible_indices_array_signle_p[dir_group] = idx
        
        # Map indices back to original order
        visible_indices_array_signle_p = order_array[visible_indices_array_signle_p]

        # Store indices
        self.visible_indices_array = np.concatenate([self.visible_indices_array, visible_indices_array_signle_p])


    def add_observeds(self, n_display_p=10_000, shape="sphere", radius=0.25, width=0.5, height=3):

        self.n_display_p = n_display_p

        for single_p in self.p_array:
            self._add_observeds_signle_p(single_p=single_p, n_display_p=self.n_display_p, shape=shape, radius=radius, width=width, height=height)


    def _add_observeds_signle_p(self, single_p, n_display_p=10_000, shape="sphere", radius=0.25, width=0.5, height=3):
        # Add point single_p to points_array to visualise single_p

        if shape == "sphere":
            # Generate random spherical coordinates
            theta = np.random.uniform(0, 2*np.pi, n_display_p)
            phi = np.random.uniform(0, np.pi, n_display_p)

            # Convert spherical coordinates to cartesian coordinates
            x_values = single_p[0] + radius * np.sin(phi) * np.cos(theta)
            y_values = single_p[1] + radius * np.sin(phi) * np.sin(theta)
            z_values = single_p[2] + radius * np.cos(phi)

            # Stack the cartesian coordinates onto the existing points_array
            display_p_array = np.column_stack((x_values, y_values, z_values, np.nan*np.ones_like(x_values), np.nan*np.ones_like(x_values), np.nan*np.ones_like(x_values)))
            self.points_array = np.vstack((self.points_array, display_p_array))

        elif shape == "cuboid":
            # Define the boundaries of the cuboid
            x_min = single_p[0] - width / 2
            x_max = single_p[0] + width / 2
            y_min = single_p[1] - width / 2
            y_max = single_p[1] + width / 2
            z_min = single_p[2] - height
            z_max = single_p[2]

            # Number of points on the top and side surface
            n_top_surface_p = int(n_display_p * 0.2)
            n_side_surface_p = n_display_p - n_top_surface_p  # Adjust ratio as needed

            # Generate points on the top surface
            x_top_surface = np.random.uniform(x_min, x_max, n_top_surface_p)
            y_top_surface = np.random.uniform(y_min, y_max, n_top_surface_p)
            z_top_surface = np.full(n_top_surface_p, z_max)

            # Generate points on the side surfaces
            x_side_surface = np.concatenate([
                np.full(n_side_surface_p // 4, x_min),
                np.full(n_side_surface_p // 4, x_max),
                np.random.uniform(x_min, x_max, n_side_surface_p // 2)
            ])
            y_side_surface = np.concatenate([
                np.random.uniform(y_min, y_max, n_side_surface_p // 2),
                np.random.uniform(y_min, y_max, n_side_surface_p // 2),
            ])
            z_side_surface = np.random.uniform(z_min, z_max, n_side_surface_p)

            # Combine points from both surfaces
            x_values = np.concatenate([x_top_surface, x_side_surface])
            y_values = np.concatenate([y_top_surface, y_side_surface])
            z_values = np.concatenate([z_top_surface, z_side_surface])

            # Stack the cartesian coordinates onto the existing points_array
            display_p_array = np.column_stack((x_values, y_values, z_values, np.nan*np.ones_like(x_values), np.nan*np.ones_like(x_values), np.nan*np.ones_like(x_values)))
            self.points_array = np.vstack((self.points_array, display_p_array))

        # Update the total number of points
        self.n_points_total += n_display_p

        # Set include_p to True
        self.include_p = True


    def colour(self, background_colour="original", observed_colour=True, observer_colour=True):
        # Colour points based on visibility

        # Define functions to perform gamma expansion/compression
        gamma_expansion = lambda c_srgb: c_srgb / 12.92 if c_srgb <= 0.04045 else ((c_srgb + 0.055) / 1.055) ** 2.4
        gamma_compression = lambda c_linear: 12.92 * c_linear if c_linear <= 0.0031308 else 1.055 * (c_linear ** (1 / 2.4)) - 0.055

        vec_gamma_expansion = np.vectorize(gamma_expansion)
        vec_gamma_compression = np.vectorize(gamma_compression)

        # Calculate number of points without p
        if self.include_p:
            n_points_wo_p = self.n_points_total - self.n_display_p * self.n_p
        else:
            n_points_wo_p = self.n_points_total

        if background_colour == "grey_scale":

            self.points_array[:n_points_wo_p, 3] = self.points_array[:n_points_wo_p, 3] / 65535
            self.points_array[:n_points_wo_p, 4] = self.points_array[:n_points_wo_p, 4] / 65535
            self.points_array[:n_points_wo_p, 5] = self.points_array[:n_points_wo_p, 5] / 65535

            self.points_array[:n_points_wo_p, 3:] = vec_gamma_expansion(self.points_array[:n_points_wo_p, 3:])
            
            y_linear = 0.2126 * self.points_array[:n_points_wo_p, 3] + 0.7152 * self.points_array[:n_points_wo_p, 4] + 0.0722 * self.points_array[:n_points_wo_p, 5]
            y_linear = y_linear / np.max(y_linear)

            y_srgb = vec_gamma_compression(y_linear)
            y_srgb = y_srgb * 65535

            self.points_array[:n_points_wo_p, 3:] = y_srgb.reshape(-1, 1)
        
        elif background_colour == "grey":

            self.points_array[:n_points_wo_p, 3:] = [30000, 30000, 30000]
        
        if observer_colour:
            # Colour visible points with red
            colour_list = []
            if self.n_p == 1:
                colour_list = [(65535, 0, 0)]
            else:
                for i in range(1, self.n_p + 1):
                    ratio = (i - 1) / (self.n_p - 1)
                    if ratio < 0.5:
                        r = int(65535 * ratio * 2)
                        g = 65535
                    else:
                        r = 65535
                        g = int(65535 * (1 - ratio) * 2)
                    b = 0
                    colour_list.append((r, g, b))

            for visible_count in np.unique(self.visible_count_array):
                
                if visible_count > 0:
                    # Get indices for visible points that meet the criteria
                    visible_indices = np.where(self.visible_count_array == visible_count)[0]

                    # Ensure Zg >= 0 for these points before coloring them
                    for idx in visible_indices:
                        if self.points_array[idx, 2] >= 0:  # Check if Zg (z value) is non-negative
                            self.points_array[idx, 3:] = colour_list[visible_count - 1]
                        else:
                            # If Zg < 0, do not color the point red
                            # Optionally, color them black or leave as is
                            # self.points_array[idx, 3:] = (0, 0, 0)  # Example to color them black
                            pass
            
        if observed_colour:
            # Colour observed with blue
            if self.include_p:
                self.points_array[n_points_wo_p:, 3:6] = [0, 0, 65535]


    def export_data(self, version=0):
        # Export the modified LAS file

        # Create a new LAS object
        new_public_header = self.public_header_list[-1]
        new_public_header.points_count = self.n_points_total

        new_las = laspy.LasData(new_public_header)
        new_las.points['x'] = self.points_array[:, 0]
        new_las.points['y'] = self.points_array[:, 1]
        new_las.points['z'] = self.points_array[:, 2]
        new_las.points['red'] = self.points_array[:, 3]
        new_las.points['green'] = self.points_array[:, 4]
        new_las.points['blue'] = self.points_array[:, 5]

        # Save the modified LAS file
        new_file_name = ''.join(map(str, self.area_names))
        new_file_path = r"Area" + new_file_name + r"_random_1_" + str(self.sample_ratio) + r"_ver" + str(version) + r".laz"
        new_las.write(new_file_path)