import os
import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, render_template, jsonify, request
from pyngrok import ngrok
from natsort import natsorted
# Initialize Flask app
app = Flask(__name__)

# Set the Ngrok auth token (ensure you have your token here)
# ngrok.set_auth_token("2rz05j4SFJLrSt48IWnNIXSmWgd_6vjhry8PfRZjHQRDgARYQ") # iitj ID
ngrok.set_auth_token("_") # AU ID


# Open an HTTP tunnel on the specified port
PORT = 8050
public_url = ngrok.connect(PORT)
print(f" * Ngrok tunnel \"{public_url}\" -> http://127.0.0.1:{PORT}")

# Function to get all folder names in the static directory
def get_folders():
    static_dir = app.static_folder  # Get path to the static folder
    return [f for f in os.listdir(static_dir) if os.path.isdir(os.path.join(static_dir, f))]

# Route to display the homepage with a dropdown and images
@app.route('/')
def index():
    folders = get_folders()  # Get all folders in the static directory
    return render_template('index.html', folders=folders)


@app.route('/get-images')
def get_images():
    folder = request.args.get('folder')
    image_dir = os.path.join('static', folder)
    images = [f"/static/{folder}/{img}" for img in natsorted(os.listdir(image_dir))]
    return jsonify(images)



# Initialize Firebase Admin SDK
cred = credentials.Certificate(os.environ.get("FIREBASE_KEY", "firebase_key.json"))  # Path to your Firebase service account key
firebase_admin.initialize_app(cred)
# Initialize Firestore client
db = firestore.client()


@app.route('/submit', methods=['POST'])
def submit():
    try:
        # Parse the JSON data from the request
        data = request.json

        # Print the received data in the backend terminal
        print("Received Data:")
        print(f"User Name: {data.get('userName')}")
        print(f"Folder: {data.get('folder')}")
        print(f"File: {data.get('file')}")
        print(f"Zoom Value: {data.get('zoomValue')}")
        print(f"Confidence Value: {data.get('confidenceValue')}")
        print(f"Real/Fake Value: {data.get('realFakeValue')}")
        print(f"Progress Value: {data.get('progressValue')}")
        print(f"Slider Value: {data.get('realFakeValueSlider')}")

        # Prepare data for Firestore document
        document_data = {
            'user_name': data.get('userName'),
            'folder': data.get('folder'),
            'file': data.get('file'),
            'zoom_value': data.get('zoomValue'),
            'confidence_value': data.get('confidenceValue'),
            'real_fake_value': data.get('realFakeValue'),
            'progress_value': data.get('progressValue'),
            'realFakeValueSlider': data.get('realFakeValueSlider')
        }

        # Store the data in Firestore
        # Create a new document in a collection named "submissions"
        db.collection('submissions').add(document_data)
        print("Data written to Firestore Database")

        return jsonify({"message": "Data received successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(port=PORT, debug=True, use_reloader=False)


