import pandas as pd
import random

# Read input CSV
input_file = "image_labels.csv"
output_file = "submissions.csv"

# Define prefix list
prefix_list = ["Definitely", "Probably"]

# Read CSV
df = pd.read_csv(input_file)

# Get total rows for percentage calculation
total_rows = len(df)

# Create new dataframe for submission
submission_data = {
    "User Name": ["temp"] * total_rows,
    "Folder": ["" for _ in range(total_rows)],
    "File": df["Filename"].tolist(),
    "Zoom Value": ["" for _ in range(total_rows)],
    "Confidence Value": ["" for _ in range(total_rows)],
    "Real/Fake Value": ["" for _ in range(total_rows)],
    "Progress Value": [f"{((i+1)/total_rows)*100:.3f}%" for i in range(total_rows)],
    "RealFakeValueSlider": [f"{random.choice(prefix_list)} {label}" for label in df["Label"]]
}

# Create DataFrame
submission_df = pd.DataFrame(submission_data)

# Save to CSV
submission_df.to_csv(output_file, index=False)

print(f"Successfully created {output_file}")
