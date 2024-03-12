import numpy as np
import laspy
import tqdm

class Visibility:
    def __init__(self, area_names, sampling="random", sample_ratio=1000):
        # Initialize Visibility object

        self.area_names = area_names if isinstance(area_names, list) else [area_names]
        self.sampling = sampling
        self.sample_ratio = sample_ratio
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


    def import_data(self, chunk_size=1000000):
        # Import data from LAS file

        # Open LAS file
        for area_name in self.area_names:

            file_path = self.generate_file_path(area_name)
            with laspy.open(file_path, mode='r') as las:
                self.public_header_list.append(las.header)
                n_points = las.header.point_count

                file_points_array = np.full((n_points, 6), np.nan)
                current_index = 0
                for points in tqdm.tqdm(las.chunk_iterator(chunk_size)):
                    file_points_array[current_index:current_index+chunk_size, 0] = points['x']
                    file_points_array[current_index:current_index+chunk_size, 1] = points['y']
                    file_points_array[current_index:current_index+chunk_size, 2] = points['z']
                    file_points_array[current_index:current_index+chunk_size, 3] = points['red']
                    file_points_array[current_index:current_index+chunk_size, 4] = points['green']
                    file_points_array[current_index:current_index+chunk_size, 5] = points['blue']
                    current_index += chunk_size

                # Update points_array and n_points_total
                self.points_array = np.vstack((self.points_array, file_points_array))
                self.n_points_total += n_points


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


    def add_observeds(self, n_display_p=10_000, radius=0.25):

        self.n_display_p = n_display_p

        for single_p in self.p_array:
            self._add_observeds_signle_p(single_p=single_p, n_display_p=self.n_display_p, radius=radius)


    def _add_observeds_signle_p(self, single_p, n_display_p=10_000, radius=0.25):
        # Add point single_p to points_array to visualise single_p

        # Generate random spherical coordinates
        theta = np.random.uniform(0, 2*np.pi, n_display_p)
        phi = np.random.uniform(0, np.pi, n_display_p)

        # Convert spherical coordinates to cartesian coordinates
        x = single_p[0] + radius * np.sin(phi) * np.cos(theta)
        y = single_p[1] + radius * np.sin(phi) * np.sin(theta)
        z = single_p[2] + radius * np.cos(phi)

        # Stack the cartesian coordinates onto the existing points_array
        display_p_array = np.column_stack((x, y, z, np.nan*np.ones_like(x), np.nan*np.ones_like(x), np.nan*np.ones_like(x)))
        self.points_array = np.vstack((self.points_array, display_p_array))

        # Update the total number of points
        self.n_points_total += n_display_p

        # Set include_p to True
        self.include_p = True


    def colour(self, colour="original"):
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

        if colour == "grey_scale":

            self.points_array[:n_points_wo_p, 3] = self.points_array[:n_points_wo_p, 3] / 65535
            self.points_array[:n_points_wo_p, 4] = self.points_array[:n_points_wo_p, 4] / 65535
            self.points_array[:n_points_wo_p, 5] = self.points_array[:n_points_wo_p, 5] / 65535

            self.points_array[:n_points_wo_p, 3:] = vec_gamma_expansion(self.points_array[:n_points_wo_p, 3:])
            
            y_linear = 0.2126 * self.points_array[:n_points_wo_p, 3] + 0.7152 * self.points_array[:n_points_wo_p, 4] + 0.0722 * self.points_array[:n_points_wo_p, 5]
            y_linear = y_linear / np.max(y_linear)

            y_srgb = vec_gamma_compression(y_linear)
            y_srgb = y_srgb * 65535

            self.points_array[:n_points_wo_p, 3:] = y_srgb.reshape(-1, 1)
        
        elif colour == "grey":

            self.points_array[:n_points_wo_p, 3:] = [30000, 30000, 30000]
            
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


# if __name__ == "__main__":
#     file_path = input("Enter the file path: ")
#     randomness = int(input("Enter the randomness: "))

#     file_path = file_path.replace(".laz", "_1_" + str(randomness) + "_random.laz")

#     lidar_processor = LidarProcessor(file_path)
#     lidar_processor.load_data()