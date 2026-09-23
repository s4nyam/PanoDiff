import os
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase Admin SDK
cred = credentials.Certificate(os.environ.get("FIREBASE_KEY", "firebase_key.json"))  # Path to your Firebase service account key
firebase_admin.initialize_app(cred)

# Initialize Firestore client
db = firestore.client()

def delete_entries_by_username(user_name):
    try:
        if not user_name:
            print("Username is required.")
            return

        # Query Firestore for all documents with the given username
        docs = db.collection('submissions').where('user_name', '==', user_name).stream()

        # Delete each document found
        deleted_count = 0
        for doc in docs:
            db.collection('submissions').document(doc.id).delete()
            deleted_count += 1

        if deleted_count == 0:
            print(f"No entries found for user: {user_name}")
        else:
            print(f"Successfully deleted {deleted_count} entries for user: {user_name}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Specify the username to delete
username_to_delete = "temp"  # Replace with the username you want to delete
delete_entries_by_username(username_to_delete)
