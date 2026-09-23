import os
import csv
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase Admin SDK
cred = credentials.Certificate(os.environ.get("FIREBASE_KEY", "firebase_key.json"))  # Path to your Firebase service account key
firebase_admin.initialize_app(cred)

# Initialize Firestore client
db = firestore.client()

# Function to download Firestore data as CSV
def download_firestore_to_csv():
    try:
        # Query all documents from the 'submissions' collection
        submissions_ref = db.collection('submissions')
        submissions = submissions_ref.stream()

        # Define the CSV file name
        csv_file_name = 'submissions.csv'

        # Open the CSV file for writing
        with open(csv_file_name, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            # Write the header row
            writer.writerow([
                'User Name', 'Folder', 'File', 'Zoom Value', 'Confidence Value',
                'Real/Fake Value', 'Progress Value', 'RealFakeValueSlider'
            ])

            # Write each document as a row in the CSV
            for submission in submissions:
                submission_data = submission.to_dict()
                writer.writerow([
                    submission_data.get('user_name', ''),
                    submission_data.get('folder', ''),
                    submission_data.get('file', ''),
                    submission_data.get('zoom_value', ''),
                    submission_data.get('confidence_value', ''),
                    submission_data.get('real_fake_value', ''),
                    submission_data.get('progress_value', ''),
                    submission_data.get('realFakeValueSlider', '')
                ])

        print(f"Data successfully written to {csv_file_name}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Run the function to download the data
download_firestore_to_csv()