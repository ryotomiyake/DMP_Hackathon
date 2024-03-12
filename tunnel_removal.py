import laspy

def color_negative_z_red_points_black(input_file, output_file):
    # Open the LAS file
    with laspy.open(input_file) as infile:
        las = infile.read()
        
        # Assuming 'red', 'green', 'blue' are stored in standard 16-bit format
        red_points_mask = (las.red == 65535) & (las.green == 0) & (las.blue == 0)
        negative_z_mask = las.z < 0
        
        # Combine the masks to find red points with negative Z
        target_points_mask = red_points_mask & negative_z_mask
        
        # Color these points black
        las.red[target_points_mask] = 0
        las.green[target_points_mask] = 0
        las.blue[target_points_mask] = 0

        # Save the modified LAS file
        las.write(output_file)

def main():
    input_file = "Area0123_random_1_100_ver0.laz"
    output_file = "Modified_Area0123_random_1_100_ver0.laz"
    color_negative_z_red_points_black(input_file, output_file)
    print("Modification completed. The output file is:", output_file)

if __name__ == "__main__":
    main()
