import cv2
import os
from tqdm import tqdm

# Define paths
hr_folder = 'HR'  # Folder containing high-resolution images
lq2_folder = 'LQ2'  # Folder to save images reduced by 2x
lq4_folder = 'LQ4'  # Folder to save images reduced by 4x

# Create output directories if they do not exist
os.makedirs(lq2_folder, exist_ok=True)
os.makedirs(lq4_folder, exist_ok=True)

# Process images
for image_name in tqdm(os.listdir(hr_folder), desc="Processing Images"):
    # Construct full image path
    hr_image_path = os.path.join(hr_folder, image_name)
    
    # Read the image
    image = cv2.imread(hr_image_path)
    if image is None:
        print(f"Failed to load {image_name}. Skipping.")
        continue

    # Get original dimensions
    height, width = image.shape[:2]

    # Reduce dimensions by 2x and 4x
    lq2_image = cv2.resize(image, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
    lq4_image = cv2.resize(image, (width // 4, height // 4), interpolation=cv2.INTER_AREA)

    # Save the resized images
    lq2_image_path = os.path.join(lq2_folder, image_name)
    lq4_image_path = os.path.join(lq4_folder, image_name)
    
    cv2.imwrite(lq2_image_path, lq2_image)
    cv2.imwrite(lq4_image_path, lq4_image)

print("Image resizing completed!")
