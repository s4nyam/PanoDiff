import os
import shutil
from tqdm import tqdm

# Define the source and destination directories
source_folder = "merged"

# Get all files in the merged folder
files_in_merged = [file for file in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, file))]

# Process each file
for file in tqdm(files_in_merged, desc="Processing files"):
    file_path = os.path.join(source_folder, file)
    
    # Check if the file name contains a suffix (e.g., "_seed0")
    if "_" in file and file.endswith(".png"):
        # Extract the folder name from the suffix
        file_name, file_ext = os.path.splitext(file)
        folder_suffix = file_name.split("_")[-3]  # Extracts the part after the last '_'
        
        # Define the destination folder based on the suffix
        destination_folder = folder_suffix
        
        # Ensure the destination folder exists
        os.makedirs(destination_folder, exist_ok=True)
        
        # Move the file to the corresponding folder
        shutil.move(file_path, os.path.join(destination_folder, file))

print(f"Files have been reorganized into their respective folders.")
