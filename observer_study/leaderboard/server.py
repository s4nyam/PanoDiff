import os
import csv
import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, render_template, jsonify
from pyngrok import ngrok
import re
from sklearn.metrics import roc_curve

# Initialize Flask app
app = Flask(__name__)

# Set the Ngrok auth token
ngrok.set_auth_token("_") # rj2298 ID

# Open an HTTP tunnel on the specified port
PORT = 8041
public_url = ngrok.connect(PORT)
print(f" * Ngrok tunnel \"{public_url}\" -> http://127.0.0.1:{PORT}")

# Initialize Firebase Admin SDK
cred = credentials.Certificate(os.environ.get("FIREBASE_KEY", "firebase_key.json"))  # Path to your Firebase service account key
firebase_admin.initialize_app(cred)

# Initialize Firestore client
db = firestore.client()

# Path to CSV file in the current directory
CSV_FILE_PATH = os.path.join(os.getcwd(), "image_labels.csv")

def load_csv_labels():
    """
    Load labels from the CSV file.
    """
    if not os.path.exists(CSV_FILE_PATH):
        print(f"CSV file not found at {CSV_FILE_PATH}")
        return {}

    label_map = {}
    with open(CSV_FILE_PATH, mode="r") as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            label_map[row["Filename"]] = row["Label"]
    return label_map

@app.route('/')
def index():
    """
    Render the main dashboard HTML.
    """
    return render_template('index.html')


# Define certainty values based on your specifications
certainty_map = {
    "Definitely Real": 100,
    "Probably Real": 50,
    "Unsure": 0,
    "Probably Fake": 50,
    "Definitely Fake": 100
}


@app.route('/get-data')
def get_data():
    """
    Fetch data from Firestore and return it as JSON.
    """
    try:
        # Load label map from CSV
        label_map = load_csv_labels()

        # Query all documents from the "submissions" collection
        submissions_ref = db.collection('submissions')
        docs = submissions_ref.stream()

        # Prepare leaderboard data
        user_scores = {}
        user_labels = {}
        user_probabilities = {}
        for doc in docs:
            submission = doc.to_dict()
            user_name = submission.get("user_name", "Unknown")
            file_name = submission.get("file")
            predicted_label = submission.get("realFakeValueSlider")
            original_label = label_map.get(file_name)
            cleaned_predicted_label = re.sub(r'^\W+|\W+$', '', predicted_label)

            if not user_scores.get(user_name):
                user_scores[user_name] = {
                    "correct": 0, "total": 0, "certainty_sum": 0,
                    "true_positives": 0, "false_positives": 0, "false_negatives": 0, "true_negatives": 0
                }
                user_labels[user_name] = []
                user_probabilities[user_name] = []

            if original_label is not None:
                user_scores[user_name]["total"] += 1
                if original_label in cleaned_predicted_label:
                    user_scores[user_name]["correct"] += 1
                    if original_label == "Fake":
                        user_scores[user_name]["true_positives"] += 1
                    else:
                        user_scores[user_name]["true_negatives"] += 1
                else:
                    if original_label == "Fake":
                        user_scores[user_name]["false_negatives"] += 1
                    else:
                        user_scores[user_name]["false_positives"] += 1
                user_scores[user_name]["certainty_sum"] += certainty_map.get(predicted_label, 0)

                # Collect data for ROC curve
                user_labels[user_name].append(1 if original_label == "Fake" else 0)
                user_probabilities[user_name].append(certainty_map.get(predicted_label, 0) / 100.0)

        # Calculate leaderboard metrics
        leaderboard = []
        for user, scores in user_scores.items():
            precision = (scores["true_positives"] / (scores["true_positives"] + scores["false_positives"])) if (scores["true_positives"] + scores["false_positives"]) > 0 else 0
            recall = (scores["true_positives"] / (scores["true_positives"] + scores["false_negatives"])) if (scores["true_positives"] + scores["false_negatives"]) > 0 else 0
            accuracy = ((scores["true_positives"] + scores["true_negatives"]) / (scores["true_positives"] + scores["true_negatives"] + scores["false_positives"] + scores["false_negatives"])) if (scores["true_positives"] + scores["true_negatives"] + scores["false_positives"] + scores["false_negatives"]) > 0 else 0
            print("{} Original Labels -: {}".format(user, user_labels[user]))
            print("{} Probabilities -: {}".format(user, user_probabilities[user]))

            # Calculate ROC curve for each user
            fpr, tpr, _ = roc_curve(user_labels[user], user_probabilities[user])
            roc_points = [{"fpr": f, "tpr": t} for f, t in zip(fpr, tpr)]
            

            leaderboard.append({
                "user": user,
                "accuracy": round((scores["correct"] / scores["total"]) * 100, 2) if scores["total"] > 0 else 0,
                "certainty": round(scores["certainty_sum"] / scores["total"], 2) if scores["total"] > 0 else 0,
                "precision": round(precision * 100, 2),
                "recall": round(recall * 100, 2),
                "calculated_accuracy": round(accuracy * 100, 2),
                "roc_points": roc_points,
                "tp": scores["true_positives"],
                "tn": scores["true_negatives"],
                "fp": scores["false_positives"],
                "fn": scores["false_negatives"]
            })

        leaderboard.sort(key=lambda x: x["accuracy"], reverse=True)
        return jsonify(leaderboard), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=PORT, debug=True, use_reloader=False)