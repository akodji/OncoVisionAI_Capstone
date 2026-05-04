# OncoVision AI

An explainable breast cancer classification web app built with EfficientNetB0, Flask, and Grad-CAM / SHAP visualisations.

**Live app:** https://onco-vision-ai.onrender.com/
**GitHub:** https://github.com/akodji/OncoVisionAI_Capstone

---

## Project Files

```
website/
├── app.py                      ← Flask backend (API, auth, database, model inference)
├── index.html                  ← Landing page
├── analyze.html                ← Image upload and classification results
├── about.html                  ← About the project and model performance
├── analytics.html              ← Clinician analytics dashboard
├── history.html                ← Past prediction history
├── profile.html                ← User profile and settings
├── audit.html                  ← Admin audit log
├── style.css                   ← Shared styles (light + dark mode)
├── darkmode.js                 ← Dark mode toggle, mobile nav, auth-aware links
├── favicon.svg                 ← App icon
├── save_shap_background.py     ← Run once in Colab to generate shap_background.npy
├── check_db.py                 ← Utility to inspect the SQLite database
├── requirements.txt            ← Python dependencies
├── runtime.txt                 ← Python version (3.11.11)
├── gunicorn.conf.py            ← Gunicorn production server config
└── Procfile                    ← Render deployment entry point
```

**Also required (provided in `resources/`, not in this folder):**
- `efficientnet_stage2_best.keras` — trained model weights
- `shap_background.npy` — SHAP background array

---

## Local Setup

### 1. Python version

Requires **Python 3.11**. Check with `python --version`.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Key packages: Flask 3.1, TensorFlow 2.18, Keras 3.8, SHAP 0.46, OpenCV (headless), bcrypt, ReportLab, scikit-learn, python-dotenv.

### 3. Place required files

Copy these into the same folder as `app.py`:
- `efficientnet_stage2_best.keras`
- `shap_background.npy`

### 4. Set environment variables

Create a `.env` file next to `app.py`:

```
SECRET_KEY=any-long-random-string
FLASK_ENV=development
```

For password-reset email (optional):
```
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-gmail-app-password
APP_BASE_URL=http://127.0.0.1:5000
```

### 5. Run

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.
Health check: `http://127.0.0.1:5000/api/health`

---

## Production Deployment (Render)

The `Procfile` and `gunicorn.conf.py` are configured for Render. Set the same environment variables in your Render dashboard. The model and SHAP background files must be present in the working directory.

---

## Important Notes

- **Classification threshold:** `0.55` — optimised on the validation set. Do not change.
- **Model:** EfficientNetB0 fine-tuned on the BreaKHis breast histopathology dataset.
- **Explainability:** Grad-CAM heatmaps and SHAP overlays are generated per prediction.
- **Demo mode:** The frontend shows a sample result if the backend is unreachable.
- **Database:** `breastai.db` is a SQLite file created automatically on first run.
- **SHAP background:** If `shap_background.npy` is missing, SHAP is skipped but Grad-CAM still works.
- **Do not commit `.env`** — it contains your secret key and email credentials.
