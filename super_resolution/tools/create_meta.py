import os
import cv2

def generate_meta_info(source_directory, output_file="meta_info.txt"):
    """
    Generates a meta information file containing image filenames and their sizes.

    Args:
        source_directory (str): Path to the directory containing images.
        output_file (str): Path to save the meta information file.
    """
    # Open the output file in write mode
    with open(output_file, "w") as file:
        for image_name in os.listdir(source_directory):
            # Construct the full image path
            image_path = os.path.join(source_directory, image_name)
            
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load {image_name}. Skipping.")
                continue
            
            # Get the dimensions of the image
            height, width, channels = image.shape
            
            # Write the metadata to the file
            file.write(f"{image_name} ({height},{width},{channels})\n")

    print(f"Meta information saved to {output_file}")

# Example usage
source_dir = "images/HR"  # Replace with the path to your source directory
output_txt_file = "meta_info.txt"  # Name of the output .txt file
generate_meta_info(source_dir, output_file=output_txt_file)
