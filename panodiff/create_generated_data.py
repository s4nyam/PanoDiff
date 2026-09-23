import os
from PIL import Image
from tqdm import tqdm

def convert_and_rename_jpegs(source_folder, destination_folder):
    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)
    
    # Gather all JPEG files excluding "grid.jpeg"
    jpeg_files = []
    for root, _, files in os.walk(source_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')) and file.lower() != "grid.jpeg":
                jpeg_files.append(os.path.join(root, file))
    
    # Initialize a counter for file renaming
    file_counter = 1

    # Process the files with a single progress bar
    for source_path in tqdm(jpeg_files, desc="Processing Images", unit="file"):
        try:
            with Image.open(source_path) as img:
                # Ensure the image is in RGB mode
                img = img.convert("RGB")
                
                # Generate the new file name
                new_name = f"generated_{file_counter:06d}.png"
                destination_path = os.path.join(destination_folder, new_name)
                
                # Save the image as PNG
                img.save(destination_path, "PNG")
                
                # Increment the counter
                file_counter += 1
        except Exception as e:
            print(f"Error processing {source_path}: {e}")

if __name__ == "__main__":
    # Example usage
    source_folder = "generated_samples"
    destination_folder = "generated_dataset"
    convert_and_rename_jpegs(source_folder, destination_folder)
