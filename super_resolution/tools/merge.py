import os
import shutil
from tqdm import tqdm

# Define the destination folder
destination_folder = "merged"

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Get a list of directories that start with "seed"
current_directory = os.getcwd()
seed_folders = [folder for folder in os.listdir(current_directory) if folder.startswith("seed") and os.path.isdir(folder)]

# Loop through each seed folder
for folder in tqdm(seed_folders, desc="Processing folders"):
    folder_path = os.path.join(current_directory, folder)
    folder_suffix = folder  # The folder name to be used as the suffix
    
    # Loop through all files in the folder
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path) and file.endswith(".jpeg"):
            # Create the new file name with the suffix
            file_name, file_ext = os.path.splitext(file)
            new_file_name = f"{file_name}_{folder_suffix}{file_ext}"
            
            # Copy the file to the destination folder with the new name
            shutil.copy(file_path, os.path.join(destination_folder, new_file_name))

print(f"Files have been processed and copied to {destination_folder}.")
