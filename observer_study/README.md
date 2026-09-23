# Observer study (paper Sections 4.1–4.2)

Six dentists — three early-career (EC1–EC3, under 10 years of experience) and three experienced
(EP1–EP3, over 15 years) — each rated the same 100 real and 100 synthetic 1024 × 512 radiographs in
one session, 12 seconds per image, on a five-level scale from "definitely real" to "definitely fake".

| File | Content |
|---|---|
| `submissions.csv` | all 1200 responses, anonymised to observer codes |
| `image_labels.csv` | ground truth (real / fake) for the 200 image names |
| `app/` | the Flask application the observers used (`server.py`, `templates/index.html`), the scripts that built the image set (`create_quiz_data/`) and the Firebase upload/download helpers |
| `leaderboard/` | the leaderboard shown after the test |

The statistics of Section 4.1 are reproduced from these two CSV files by

```bash
python evaluation/observer_study/observer_stats.py observer_study
```

## Running the application

```bash
pip install flask firebase-admin
export FIREBASE_KEY=/path/to/your-service-account.json    # format: app/firebase_key.example.json
python app/server.py
```

The 200 images are not included; they can be requested from the authors. You can take the same test
on the [project page](https://s4nyam.github.io/panodiff/#test).
