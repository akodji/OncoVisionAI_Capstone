OncoVision AI — Submission README
Capstone Project, Ashesi University
Anna Kodji, BSc Computer Science

================================================================
LIVE APPLICATION
================================================================
https://onco-vision-ai.onrender.com/

GitHub Repository:
https://github.com/akodji/OncoVisionAI_Capstone

================================================================
WHAT THE APP DOES
================================================================
OncoVision AI is a web-based explainable AI tool for breast cancer
classification from histopathology images. It uses an EfficientNetB0
deep learning model fine-tuned on the BreaKHis dataset and provides
Grad-CAM heatmaps and SHAP overlays so predictions are interpretable
for clinicians.

Features:
- Image upload and binary classification (Malignant / Benign)
- Grad-CAM and SHAP explainability visualisations per prediction
- Clinician analytics dashboard
- Prediction history with edit/delete
- User accounts with secure authentication
- PDF report export
- Admin audit log
- Dark mode

================================================================
HOW TO INSTALL (Local Setup)
================================================================
Requirements: Python 3.11

1. Install dependencies:
   pip install -r requirements.txt

2. Place these two files in the same folder as app.py:
   - efficientnet_stage2_best.keras   (trained model — in resources/)
   - shap_background.npy              (SHAP background — in resources/)

3. Create a .env file in the same folder as app.py:

   SECRET_KEY=any-long-random-string
   FLASK_ENV=development

   (SMTP variables below are optional — only needed for password reset emails)
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USER=your-email@gmail.com
   SMTP_PASSWORD=your-gmail-app-password
   APP_BASE_URL=http://127.0.0.1:5000

================================================================
HOW TO RUN
================================================================
python app.py

Then open in your browser: http://127.0.0.1:5000
Health check endpoint:     http://127.0.0.1:5000/api/health

================================================================
IMPORTANT NOTES
================================================================
- Classification threshold is 0.55, optimised on the validation set.
  Do not change this value.
- The database file (breastai.db) is created automatically on first run.
- If shap_background.npy is missing, SHAP visualisation is skipped
  but Grad-CAM still works.
- Demo mode: the frontend shows a sample result if the backend is
  unreachable — useful during presentations without a running server.
- Do NOT submit your .env file — it contains your secret key and
  email credentials.

================================================================
SUBMISSION FOLDER STRUCTURE
================================================================
report/          Final project report (PDF)
presentation/    Slide deck (PDF)
demo/            Demo video
documentation/   This readme.txt
code/
  website/       Flask backend + HTML/CSS/JS frontend
  notebooks/     Model training and SHAP notebooks (Google Colab)
  ml_hosting/    ML hosting configuration files
resources/       efficientnet_stage2_best.keras + shap_background.npy
data/            breastai.db (SQLite database) + prediction_log.csv
other/           Miscellaneous files
