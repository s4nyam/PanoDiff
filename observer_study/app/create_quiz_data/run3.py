# source1 = "dataset/FakeImages" # Fake Images Dataset

# source2 = "dataset/OriginalImages" # Real Training Images Dataset

# Code steps now:

# step 1: read all images in fake images and real images directory, source1 and source 2

# step 2: use random seed throughout the file and shuffle them. Then pick 100 from source1 and 100 from source 2

# step 3: rename with alpha numeric random string and save it in destination directory.

# step 4: For all the png files you picked from real folder, put lable as "Real" and for all images picked from fake folder, put label as "Fake", This way you save a csv file that has two columns, column is is Filename and column 2 is Label The Filename column will have the filenmae that is given as alpha numeric string.

# step 5: So in the output I will have a desinatination directory which is populaed with 200 images + csv file that tells the Real and Fake label.

# import os
# import random
# import shutil
# import string
# import pandas as pd

# # Paths to source directories
# source1 = "FakeImages"  # Fake Images Dataset
# source2 = "OriginalImages"  # Real Training Images Dataset
# destination = "fullres"  # Destination directory where images will be saved
# root = "."
# # Step 1: Read all images in fake images and real images directory
# fake_images = [f for f in os.listdir(source1) if f.endswith('.png')]  # or use .jpg/.jpeg if necessary
# real_images = [f for f in os.listdir(source2) if f.endswith('.png')]  # same as above

# # Step 2: Shuffle both lists and pick 100 random images from each
# random.seed(29)  # Ensuring reproducibility
# random.shuffle(fake_images)
# random.shuffle(real_images)

# fake_images_selected = fake_images[:100]
# real_images_selected = real_images[:100]
# combined = fake_images_selected + real_images_selected
# random.shuffle(combined)
# # Step 3: Rename images with random alphanumeric string and save them in destination directory
# if not os.path.exists(destination):
#     os.makedirs(destination)

# renamed_images = []

# def generate_random_string(length=8):
#     return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

# # Move and rename images
# for idx, img in enumerate(combined):
#     random_filename = generate_random_string() + '.png'  # Change extension if needed
#     src = os.path.join(source1 if img in fake_images_selected else source2, img)
#     dst = os.path.join(destination, random_filename)
    
#     shutil.copy(src, dst)  # Copy image to destination
#     renamed_images.append((random_filename, 'Fake' if img in fake_images_selected else 'Real'))

# # Step 4: Create CSV file with filenames and labels
# df = pd.DataFrame(renamed_images, columns=['Filename', 'Label'])
# csv_file = os.path.join(root, 'image_labels.csv')
# df.to_csv(csv_file, index=False)

# print(f"Processed {len(renamed_images)} images. CSV file saved at {csv_file}.")




import os
import random
import shutil
import string
import pandas as pd

# Paths to source directories
source1 = "FakeImages"  # Fake Images Dataset
source2 = "OriginalImages_Processed_25selections_final"  # Real Training Images Dataset
destination = "OriginalImages_Processed_25selections_final_fullres"  # Destination directory where images will be saved
destination_separated = "OriginalImages_Processed_25selections_final_fullres_separated"  # Destination directory with 'fake' and 'real' subfolders
root = "."  # Root directory for CSV file

# Step 1: Read all images in fake images and real images directory
fake_images = [f for f in os.listdir(source1) if f.endswith('.png')]  # or use .jpg/.jpeg if necessary
real_images = [f for f in os.listdir(source2) if f.endswith('.png')]  # same as above

# Step 2: Shuffle both lists and pick 100 random images from each
random.seed(29)  # Ensuring reproducibility
random.shuffle(fake_images)
random.shuffle(real_images)

fake_images_selected = fake_images[:100]
real_images_selected = real_images[:100]
combined = fake_images_selected + real_images_selected
random.shuffle(combined)

# Step 3: Rename images with random alphanumeric string and save them in destination directory
if not os.path.exists(destination):
    os.makedirs(destination)

# Create the separated destination folder structure
fake_folder = os.path.join(destination_separated, "fake")
real_folder = os.path.join(destination_separated, "real")

if not os.path.exists(destination_separated):
    os.makedirs(destination_separated)
if not os.path.exists(fake_folder):
    os.makedirs(fake_folder)
if not os.path.exists(real_folder):
    os.makedirs(real_folder)

renamed_images = []

def generate_random_string(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

# Move and rename images
for idx, img in enumerate(combined):
    random_filename = generate_random_string() + '.png'  # Change extension if needed
    label = 'Fake' if img in fake_images_selected else 'Real'
    src = os.path.join(source1 if label == 'Fake' else source2, img)
    dst = os.path.join(destination, random_filename)  # Save in main destination
    shutil.copy(src, dst)
    
    # Save in separated folder
    dst_separated = os.path.join(fake_folder if label == 'Fake' else real_folder, random_filename)
    shutil.copy(src, dst_separated)
    
    renamed_images.append((random_filename, label))

# Step 4: Create CSV file with filenames and labels
df = pd.DataFrame(renamed_images, columns=['Filename', 'Label'])
csv_file = os.path.join(root, 'OriginalImages_Processed_25selections_final_fullres_separated.csv')
df.to_csv(csv_file, index=False)

print(f"Processed {len(renamed_images)} images.")
print(f"Images saved in '{destination}' and '{destination_separated}'.")
print(f"CSV file saved at {csv_file}.")
