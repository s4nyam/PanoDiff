import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import os
from PIL import Image
from tqdm import tqdm

def process_images(src_dir, dest_dir, crop_values, output_size=(1024, 512)):

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Supported image extensions
    valid_extensions = {".png", ".jpg", ".jpeg", ".JPG"}
    
    # Process each file in the source directory
    for file_name in tqdm(os.listdir(src_dir), desc="Processing images"):
        file_path = os.path.join(src_dir, file_name)
        file_extension = os.path.splitext(file_name)[1]
        
        if file_extension in valid_extensions:
            try:
                # Open the image
                with Image.open(file_path) as img:
                    # Calculate cropping box
                    top, left, bottom, right = crop_values
                    width, height = img.size
                    
                    # Define new cropping coordinates
                    crop_box = (
                        left,
                        top,
                        width - right,
                        height - bottom
                    )
                    
                    # Crop the image
                    cropped_img = img.crop(crop_box)
                    
                    # Resize the image
                    resized_img = cropped_img.resize(output_size, Image.Resampling.LANCZOS)
                    
                    # Save the image as PNG with the same name
                    dest_path = os.path.join(dest_dir, os.path.splitext(file_name)[0] + ".png")
                    resized_img.save(dest_path, "PNG")
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")


# ADLD Dataset
source_directory = WORK + "/quiz/dataset/OriginalDatasets/adld"
destination_directory = WORK + "/quiz/dataset/OriginalImages_Processed/adld_processed"
crop_top_left_bottom_right = (90, 30, 30, 170)
process_images(source_directory, destination_directory, crop_top_left_bottom_right)

# Dentex Dataset
source_directory = WORK + "/quiz/dataset/OriginalDatasets/dentex"
destination_directory = WORK + "/quiz/dataset/OriginalImages_Processed/dentex_processed"
crop_top_left_bottom_right = (20, 100, 50, 20)
process_images(source_directory, destination_directory, crop_top_left_bottom_right)


# Tufts Dataset
source_directory = WORK + "/quiz/dataset/OriginalDatasets/tufts"
destination_directory = WORK + "/quiz/dataset/OriginalImages_Processed/tufts_processed"
crop_top_left_bottom_right = (20, 30, 100, 100)
process_images(source_directory, destination_directory, crop_top_left_bottom_right)


# uspforp Dataset
source_directory = WORK + "/quiz/dataset/OriginalDatasets/uspforp"
destination_directory = WORK + "/quiz/dataset/OriginalImages_Processed/uspforp_processed"
crop_top_left_bottom_right = (30, 30, 165, 30)
process_images(source_directory, destination_directory, crop_top_left_bottom_right)