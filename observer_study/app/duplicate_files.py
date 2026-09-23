import os
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase Admin SDK
cred = credentials.Certificate(os.environ.get("FIREBASE_KEY", "firebase_key.json"))  # Path to your Firebase service account key
firebase_admin.initialize_app(cred)

# Initialize Firestore client
db = firestore.client()

def delete_duplicates_for_user(user_name):
    try:
        if not user_name:
            print("Username is required.")
            return

        # Query Firestore for all documents with the given username
        docs = db.collection('submissions').where('user_name', '==', user_name).stream()

        # Store file names seen for this user
        seen_files = set()
        duplicate_docs = []

        # Loop through the documents and identify duplicates based on the file name
        for doc in docs:
            file_name = doc.to_dict().get('file')
            if file_name in seen_files:
                # If the file has been seen before, mark this document for deletion
                duplicate_docs.append(doc.id)
            else:
                # If it's a new file for the user, add it to the seen set
                seen_files.add(file_name)

        # Delete the duplicate entries
        deleted_count = 0
        for doc_id in duplicate_docs:
            db.collection('submissions').document(doc_id).delete()
            deleted_count += 1

        if deleted_count == 0:
            print(f"No duplicate entries found for user: {user_name}")
        else:
            print(f"Successfully deleted {deleted_count} duplicate entries for user: {user_name}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Specify the username to delete duplicates
username_to_check = "sanyam"  # Replace with the username you want to check for duplicates
delete_duplicates_for_user(username_to_check)
