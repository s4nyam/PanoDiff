import os
import csv
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase Admin SDK
cred = credentials.Certificate(os.environ.get("FIREBASE_KEY", "firebase_key.json"))  # Path to your Firebase service account key
firebase_admin.initialize_app(cred)

# Initialize Firestore client
db = firestore.client()

# Function to upload CSV data to Firestore
def upload_csv_to_firestore(csv_file_path):
    try:
        # Open the CSV file for reading
        with open(csv_file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)

            # Iterate through each row in the CSV file
            for row in reader:
                # Prepare data dictionary, replacing 'null' with None (Firestore equivalent)
                data = {
                    'user_name': row.get('User Name', '') if row.get('User Name', '') != 'null' else None,
                    'folder': row.get('Folder', '') if row.get('Folder', '') != 'null' else None,
                    'file': row.get('File', '') if row.get('File', '') != 'null' else None,
                    'zoom_value': row.get('Zoom Value', '') if row.get('Zoom Value', '') != 'null' else None,
                    'confidence_value': row.get('Confidence Value', '') if row.get('Confidence Value', '') != 'null' else None,
                    'real_fake_value': row.get('Real/Fake Value', '') if row.get('Real/Fake Value', '') != 'null' else None,
                    'progress_value': row.get('Progress Value', '') if row.get('Progress Value', '') != 'null' else None,
                    'realFakeValueSlider': row.get('RealFakeValueSlider', '') if row.get('RealFakeValueSlider', '') != 'null' else None
                }

                # Add document to Firestore in 'submissions' collection
                db.collection('submissions').add(data)

        print(f"Data successfully uploaded from {csv_file_path} to Firestore.")

    except Exception as e:
        print(f"An error occurred: {e}")

# Run the function to upload data
upload_csv_to_firestore('submissions.csv')
