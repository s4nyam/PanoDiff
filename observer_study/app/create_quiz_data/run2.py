import os
import random
import shutil
from tqdm import tqdm
random.seed(10)
def pick_and_copy_images_from_each_folder(src_dir, dest_dir, num_images=25):
    # Supported image extensions
    valid_extensions = {".png", ".jpg", ".jpeg", ".JPG"}

    # Walk through the source directory
    for root, _, files in os.walk(src_dir):
        # Collect all valid images in the current folder
        valid_images = [
            os.path.join(root, file_name)
            for file_name in files
            if os.path.splitext(file_name)[1] in valid_extensions
        ]

        # Randomly select up to `num_images` from the current folder
        selected_images = random.sample(valid_images, min(len(valid_images), num_images))

        # Copy each selected image to the destination, preserving folder structure
        for img_path in tqdm(selected_images, desc=f"Processing folder: {root}"):
            # Compute relative path from source directory to the image
            relative_path = os.path.relpath(img_path, src_dir)
            dest_path = os.path.join(dest_dir, relative_path)

            # Ensure the destination folder exists
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            # Copy the image to the destination directory
            shutil.copy2(img_path, dest_path)

    print(f"Images copied successfully to {dest_dir}")


source_directory = "OriginalImages_Processed"
destination_directory = "OriginalImages_Processed_25selections"

pick_and_copy_images_from_each_folder(source_directory, destination_directory, num_images=25)

