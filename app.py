"""
OncoVision AI Flask Backend
"""

import os
import io
import re
import uuid
import base64
import sqlite3
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import numpy as np
from PIL import Image
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, session, redirect, render_template_string, send_from_directory, make_response
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from functools import wraps
import bcrypt
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
import shap
import cv2
from scipy.ndimage import gaussian_filter
from werkzeug.utils import secure_filename
from flask import send_file
from dotenv import load_dotenv

load_dotenv()

# ── CONFIGURATION ─────────────────────────────────────────────────────
MODEL_PATH   = 'efficientnet_stage2_best.keras'
IMG_SIZE     = 224
THRESHOLD    = 0.55
LAST_CONV    = 'top_conv'
SHAP_BG_PATH = 'shap_background.npy'
DB_PATH      = 'breastai.db'
SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-here-make-it-long-and-random-BreastAI2026')

# ── EMAIL CONFIGURATION ───────────────────────────────────────────────
SMTP_HOST     = os.environ.get('SMTP_HOST',     'smtp.gmail.com')
SMTP_PORT     = int(os.environ.get('SMTP_PORT', '587'))
SMTP_USER     = os.environ.get('SMTP_USER',     '')   # e.g. yourapp@gmail.com
SMTP_PASSWORD = os.environ.get('SMTP_PASSWORD', '')   # Gmail App Password
APP_BASE_URL  = os.environ.get('APP_BASE_URL',  'http://127.0.0.1:5000')

# File upload security
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
HISTORY_FOLDER = 'history_images'

# ── APP SETUP ─────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('FLASK_ENV', 'production') != 'development'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)
app.config['SESSION_COOKIE_DOMAIN'] = None
app.config['SESSION_COOKIE_PATH'] = '/'
_cors_origins = os.environ.get('ALLOWED_ORIGINS', 'http://localhost:5000,http://127.0.0.1:5000').split(',')
CORS(app,
     supports_credentials=True,
     origins=_cors_origins,
     allow_headers=['Content-Type'],
     expose_headers=['Set-Cookie'])

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri='memory://'
)

os.makedirs(HISTORY_FOLDER, exist_ok=True)

# ── DATABASE ──────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT UNIQUE NOT NULL,
            password    TEXT NOT NULL,
            full_name   TEXT NOT NULL,
            role        TEXT NOT NULL,
            institution TEXT,
            is_admin    INTEGER DEFAULT 0,
            created_at  TEXT NOT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id  TEXT UNIQUE NOT NULL,
            user_id        INTEGER,
            username       TEXT,
            patient_name   TEXT,
            filename       TEXT,
            prediction     TEXT NOT NULL,
            confidence     REAL NOT NULL,
            probability    REAL NOT NULL,
            threshold      REAL NOT NULL,
            original_image TEXT,
            gradcam_image  TEXT,
            shap_image     TEXT,
            created_at     TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Add new columns if they don't exist (migration)
    try:
        c.execute('ALTER TABLE users ADD COLUMN email TEXT')
    except sqlite3.OperationalError:
        pass

    try:
        c.execute('ALTER TABLE predictions ADD COLUMN prediction_id TEXT')
    except sqlite3.OperationalError:
        pass

    try:
        c.execute('ALTER TABLE predictions ADD COLUMN original_image TEXT')
    except sqlite3.OperationalError:
        pass

    try:
        c.execute('ALTER TABLE predictions ADD COLUMN gradcam_image TEXT')
    except sqlite3.OperationalError:
        pass

    try:
        c.execute('ALTER TABLE predictions ADD COLUMN shap_image TEXT')
    except sqlite3.OperationalError:
        pass

    try:
        c.execute('ALTER TABLE predictions ADD COLUMN notes TEXT')
    except sqlite3.OperationalError:
        pass
    
    # Create default admin user if it doesn't exist
    c.execute('SELECT COUNT(*) FROM users WHERE username=?', ('admin',))
    if c.fetchone()[0] == 0:
        default_pwd = os.environ.get('ADMIN_PASSWORD', 'BreastAI2026!')
        hashed = bcrypt.hashpw(default_pwd.encode('utf-8'), bcrypt.gensalt())
        c.execute('''
            INSERT INTO users (username, password, full_name, role, institution, is_admin, created_at)
            VALUES (?, ?, ?, ?, ?, 1, ?)
        ''', ('admin', hashed.decode('utf-8'), 'System Administrator',
              'Administrator', 'Ashesi University',
              datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    c.execute('''\
        CREATE TABLE IF NOT EXISTS audit_logs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER,
            username    TEXT,
            action      TEXT NOT NULL,
            detail      TEXT,
            ip_address  TEXT,
            created_at  TEXT NOT NULL
        )
    ''')
    c.execute('''\
        CREATE TABLE IF NOT EXISTS password_reset_tokens (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            token      TEXT UNIQUE NOT NULL,
            expires_at TEXT NOT NULL,
            used       INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ── FILE UPLOAD SECURITY ──────────────────────────────────────────────
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_file_size(file_bytes):
    return len(file_bytes) <= MAX_FILE_SIZE

# ── PASSWORD VALIDATION ───────────────────────────────────────────────
def validate_password(password):
    """Validate password strength - returns list of missing requirements"""
    errors = []
    if len(password) < 8:
        errors.append('at least 8 characters')
    if not re.search(r"[A-Z]", password):
        errors.append('one uppercase letter')
    if not re.search(r"[0-9]", password):
        errors.append('one number')
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        errors.append('one special character')
    return errors

# ── USER HELPERS ──────────────────────────────────────────────────────
def get_user(username):
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
    conn.close()
    return user

def username_exists(username):
    return get_user(username) is not None

def create_user(username, password, full_name, role, institution, email=''):
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    conn = get_db()
    conn.execute('''
        INSERT INTO users (username, password, full_name, role, institution, email, is_admin, created_at)
        VALUES (?, ?, ?, ?, ?, ?, 0, ?)
    ''', (username, hashed.decode('utf-8'), full_name, role, institution, email,
          datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()
    conn.close()

def get_user_by_email(email):
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
    conn.close()
    return user

def send_reset_email(to_email, reset_token):
    """Send a password reset email. Returns True on success, False if unconfigured or failed."""
    if not SMTP_USER or not SMTP_PASSWORD:
        return False
    reset_url = f"{APP_BASE_URL}/reset-password/{reset_token}"
    msg = MIMEMultipart('alternative')
    msg['Subject'] = 'Reset your OncoVision AI password'
    msg['From']    = f'OncoVision AI <{SMTP_USER}>'
    msg['To']      = to_email
    text = (f"You requested a password reset for your OncoVision AI account.\n\n"
            f"Click the link below to set a new password (valid for 1 hour):\n{reset_url}\n\n"
            f"If you did not request this, you can safely ignore this email.")
    html = f"""
    <div style="font-family:sans-serif;max-width:480px;margin:0 auto;padding:2rem;background:#fdf8fa;border-radius:12px;">
      <h2 style="color:#c67489;margin-bottom:0.5rem;">OncoVision AI</h2>
      <h3 style="color:#333;margin-bottom:1rem;">Password Reset Request</h3>
      <p style="color:#666;margin-bottom:1.5rem;">
        You requested a password reset. Click the button below to set a new password.
        This link is valid for <strong>1 hour</strong>.
      </p>
      <a href="{reset_url}" style="display:inline-block;background:#c67489;color:white;padding:0.85rem 2rem;border-radius:8px;text-decoration:none;font-weight:600;">
        Reset My Password
      </a>
      <p style="color:#999;font-size:0.8rem;margin-top:1.5rem;">
        If the button doesn't work, copy this link:<br>
        <a href="{reset_url}" style="color:#c67489;word-break:break-all;">{reset_url}</a>
      </p>
      <p style="color:#bbb;font-size:0.75rem;margin-top:1rem;">
        If you did not request this, you can safely ignore this email.
      </p>
    </div>"""
    msg.attach(MIMEText(text, 'plain'))
    msg.attach(MIMEText(html, 'html'))
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, to_email, msg.as_string())
        return True
    except Exception:
        return False

def check_password(username, password):
    user = get_user(username)
    if not user:
        return False
    return bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8'))

def log_prediction(user_id, username, patient_name, filename, prediction, confidence, probability,
                   original_img, gradcam_img, shap_img):
    prediction_id = str(uuid.uuid4())
    conn = get_db()
    conn.execute('''
        INSERT INTO predictions
        (prediction_id, user_id, username, patient_name, filename, prediction, confidence, 
         probability, threshold, original_image, gradcam_image, shap_image, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (prediction_id, user_id, username, patient_name or 'Unknown', filename or 'unknown',
          prediction, round(confidence, 4), round(probability, 4), THRESHOLD,
          original_img, gradcam_img, shap_img,
          datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()
    conn.close()
    return prediction_id

def get_user_logs(user_id):
    conn = get_db()
    rows = conn.execute(
        'SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC', (user_id,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_prediction_by_id(prediction_id, user_id):
    conn = get_db()
    row = conn.execute(
        'SELECT * FROM predictions WHERE prediction_id = ? AND user_id = ?',
        (prediction_id, user_id)
    ).fetchone()
    conn.close()
    return dict(row) if row else None

def update_user(user_id, full_name, role, institution, email=''):
    conn = get_db()
    conn.execute(
        "UPDATE users SET full_name=?, role=?, institution=?, email=? WHERE id=?",
        (full_name, role, institution, email, user_id)
    )
    conn.commit()
    conn.close()

def update_password(user_id, new_password):
    hashed = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
    conn = get_db()
    conn.execute("UPDATE users SET password=? WHERE id=?", (hashed.decode('utf-8'), user_id))
    conn.commit()
    conn.close()

def get_all_users():
    conn = get_db()
    rows = conn.execute('''
        SELECT u.id, u.username, u.full_name, u.role, u.institution, u.created_at,
               u.is_admin, COUNT(p.id) as prediction_count
        FROM users u
        LEFT JOIN predictions p ON u.id = p.user_id
        WHERE u.is_admin = 0
        GROUP BY u.id
        ORDER BY u.created_at DESC
    ''').fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_all_predictions():
    conn = get_db()
    rows = conn.execute('''
        SELECT p.*, u.full_name, u.role
        FROM predictions p
        LEFT JOIN users u ON p.user_id = u.id
        ORDER BY p.created_at DESC
        LIMIT 500
    ''').fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_stats():
    conn = get_db()
    total_users = conn.execute("SELECT COUNT(*) FROM users WHERE is_admin=0").fetchone()[0]
    total_preds = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    malignant   = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction='Malignant'").fetchone()[0]
    benign      = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction='Benign'").fetchone()[0]
    today       = datetime.now().strftime('%Y-%m-%d')
    today_preds = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE created_at LIKE ?", (today + '%',)
    ).fetchone()[0]
    conn.close()
    return {
        'total_users': total_users,
        'total_preds': total_preds,
        'malignant':   malignant,
        'benign':      benign,
        'today_preds': today_preds,
        'mal_pct':     round(malignant / total_preds * 100, 1) if total_preds > 0 else 0,
        'ben_pct':     round(benign / total_preds * 100, 1) if total_preds > 0 else 0,
    }

def log_audit(action, detail=None):
    """Log a user action to the audit_logs table."""
    user_id  = session.get('user_id')
    username = session.get('username', 'anonymous')
    ip       = request.remote_addr if request else None
    conn = get_db()
    conn.execute(
        'INSERT INTO audit_logs (user_id, username, action, detail, ip_address, created_at) VALUES (?,?,?,?,?,?)',
        (user_id, username, action, detail, ip, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    )
    conn.commit()
    conn.close()

def get_audit_logs(limit=200):
    conn = get_db()
    rows = conn.execute(
        'SELECT * FROM audit_logs ORDER BY created_at DESC LIMIT ?', (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            return jsonify({'error': 'Not authenticated'}), 401
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('is_admin'):
            return jsonify({'error': 'Admin access required'}), 403
        return f(*args, **kwargs)
    return decorated

# ── MODEL LOADING ─────────────────────────────────────────────────────
import tensorflow.keras.backend as K

# Limit TF thread usage to reduce memory overhead on small servers
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

def loss_fn(y_true, y_pred):
    return K.binary_crossentropy(y_true, y_pred)

print('Loading EfficientNetB0 model...')
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'loss_fn': loss_fn}, compile=False)
# No compile() needed — we only run inference, not training
print(f'Model loaded. Parameters: {model.count_params():,}')

shap_explainer = None
shap_explainer_lock = False

def get_shap_explainer():
    global shap_explainer, shap_explainer_lock
    if shap_explainer is not None:
        return shap_explainer
    if shap_explainer_lock:
        return None
    try:
        shap_explainer_lock = True
        print('Loading SHAP background images...')
        shap_background = np.load(SHAP_BG_PATH)
        shap_explainer = shap.GradientExplainer(model, shap_background)
        print('SHAP explainer ready.')
    except Exception as e:
        print(f'SHAP explainer failed to load: {e}')
        shap_explainer = None
    finally:
        shap_explainer_lock = False
    return shap_explainer

def load_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    return img

def to_model_input(img):
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# top_activation is the correct layer for EfficientNetB0 Grad-CAM.
# Its corner artifact (padding boundary effect) is fixed by zeroing the
# outermost ring of the 7x7 feature map before upscaling.
GRADCAM_LAYER = 'top_activation'
print(f'Grad-CAM layer: {GRADCAM_LAYER}')

def generate_gradcam(img_input, raw_img):
    grad_model = Model([model.input],
                       [model.get_layer(GRADCAM_LAYER).output, model.output])

    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(img_input)
        if isinstance(pred, list):
            pred = pred[0]
        loss = pred[0][0]

    grads  = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(1, 2))[0]

    # Weighted sum across channels → (7, 7) heatmap
    heatmap = tf.reduce_sum(conv_out[0] * pooled, axis=-1).numpy()

    # Standard Grad-CAM: ReLU keeps only positive contributions
    heatmap = np.maximum(heatmap, 0)

    # Percentile normalisation: clip at the 90th percentile so only the top 10%
    # of activations reach red — prevents the "everything is red" effect caused
    # by EfficientNet's 7×7 map having many uniformly high positive values
    positive = heatmap[heatmap > 0]
    if positive.size > 0:
        p90 = np.percentile(positive, 90)
        heatmap = np.clip(heatmap, 0, p90) / (p90 + 1e-10)
    else:
        heatmap = heatmap / (heatmap.max() + 1e-10)

    original_arr = np.array(raw_img.resize((IMG_SIZE, IMG_SIZE)))
    if original_arr.ndim == 2:
        original_arr = cv2.cvtColor(original_arr, cv2.COLOR_GRAY2RGB)
    elif original_arr.shape[2] == 4:
        original_arr = original_arr[:, :, :3]

    h, w       = original_arr.shape[:2]
    # Bicubic interpolation for smoother upscaling from 7×7 → 224×224
    hm_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    hm_resized = np.clip(hm_resized, 0, 1)
    # Mild smoothing to reduce blockiness
    hm_resized = gaussian_filter(hm_resized, sigma=1.5)
    hm_resized = hm_resized / (hm_resized.max() + 1e-10)

    # Apply JET colormap — standard Grad-CAM visualisation
    hm_color = cv2.applyColorMap(np.uint8(255 * hm_resized), cv2.COLORMAP_JET)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)

    # Overlay: standard 60/40 blend matching Colab style
    overlay = cv2.addWeighted(original_arr.astype(np.uint8), 0.6,
                              hm_color.astype(np.uint8), 0.4, 0)

    return {
        'original': base64.b64encode(cv2.imencode('.png', original_arr)[1]).decode(),
        'heatmap':  base64.b64encode(cv2.imencode('.png', hm_color)[1]).decode(),
        'overlay':  base64.b64encode(cv2.imencode('.png', overlay)[1]).decode(),
    }

def generate_shap(img_input, raw_img):
    shap_values = get_shap_explainer().shap_values(img_input)

    # GradientExplainer returns either:
    #   - a list of arrays (one per output node) → shape each: (1, H, W, C) or (1, H, W, C, 1)
    #   - a plain array of shape (1, H, W, C) or (1, H, W, C, 1)
    # Match Colab: shap_values[row][:, :, :, 0] for shape (H, W, C, 1)
    sv = shap_values[0] if isinstance(shap_values, list) else shap_values
    sv = np.array(sv)

    # Remove batch dimension (index 0 = our single image)
    if sv.ndim >= 4 and sv.shape[0] == 1:
        sv = sv[0]   # → (H, W, C) or (H, W, C, 1)

    # Drop trailing output dimension if present, matching Colab's [:, :, :, 0]
    if sv.ndim == 4 and sv.shape[-1] == 1:
        sv = sv[..., 0]   # → (H, W, C)

    # sv is now (H, W, C) = (224, 224, 3)

    # Aggregate across RGB channels with mean absolute value — matches Colab exactly
    shap_agg = np.mean(np.abs(sv), axis=-1)          # (224, 224)

    # Smooth with same sigma as Colab
    shap_agg = gaussian_filter(shap_agg, sigma=3)

    # Normalise by dividing by max — matches Colab exactly
    if shap_agg.max() > 0:
        shap_agg = shap_agg / shap_agg.max()

    shap_heatmap = cv2.applyColorMap(np.uint8(255 * shap_agg), cv2.COLORMAP_JET)
    shap_heatmap = cv2.cvtColor(shap_heatmap, cv2.COLOR_BGR2RGB)

    # Prepare original image — normalise to [0, 255] uint8, matching Colab's display_img
    original_arr = np.array(raw_img.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32)
    if original_arr.ndim == 2:
        original_arr = cv2.cvtColor(original_arr, cv2.COLOR_GRAY2RGB)
    elif original_arr.shape[2] == 4:
        original_arr = original_arr[:, :, :3]
    original_arr = original_arr - original_arr.min()
    if original_arr.max() > 0:
        original_arr = original_arr / original_arr.max()
    original_arr = (original_arr * 255).astype(np.uint8)

    h, w = original_arr.shape[:2]
    shap_heatmap = cv2.resize(shap_heatmap, (w, h))

    overlay = cv2.addWeighted(original_arr, 0.6, shap_heatmap.astype(np.uint8), 0.4, 0)

    return {
        'original': base64.b64encode(cv2.imencode('.png', original_arr)[1]).decode(),
        'heatmap':  base64.b64encode(cv2.imencode('.png', shap_heatmap)[1]).decode(),
        'overlay':  base64.b64encode(cv2.imencode('.png', overlay)[1]).decode(),
    }

def save_image_base64(b64_data, prefix='img'):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{timestamp}_{unique_id}_{prefix}.png"
    filepath = os.path.join(HISTORY_FOLDER, filename)
    
    img_data = base64.b64decode(b64_data)
    with open(filepath, 'wb') as f:
        f.write(img_data)
    
    return filename

# ── AUTH ROUTES ───────────────────────────────────────────────────────
@app.route('/login', methods=['GET', 'POST'])
@limiter.limit('10 per minute', methods=['POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        if check_password(username, password):
            user = get_user(username)
            session['logged_in'] = True
            session['user_id']   = user['id']
            session['username']  = user['username']
            session['full_name'] = user['full_name']
            session['role']      = user['role']
            session['is_admin']  = user['is_admin']
            session.permanent    = True
            log_audit('LOGIN', f'User logged in from {request.remote_addr}')
            return jsonify({'success': True, 'is_admin': bool(user['is_admin'])})
        else:
            return jsonify({'error': 'Invalid username or password'}), 401
    
    return render_template_string('''<!DOCTYPE html>
<html>
<head><title>Sign In — OncoVision AI</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg"/>
<style>
  body { font-family: sans-serif; background: #fdf8fa; margin: 0; display: flex; align-items: center; justify-content: center; min-height: 100vh; }
  .card { background: white; border-radius: 12px; padding: 2.5rem; max-width: 380px; width: 100%; box-shadow: 0 4px 6px rgba(0,0,0,0.07); }
  h1 { font-size: 1.8rem; margin-bottom: 0.5rem; }
  p { color: #999; font-size: 0.9rem; margin-bottom: 2rem; }
  input { width: 100%; padding: 0.75rem; margin-bottom: 1rem; border: 1px solid #ddd; border-radius: 6px; font-size: 0.95rem; box-sizing: border-box; }
  input:focus { outline: none; border-color: #c67489; }
  button { width: 100%; padding: 0.85rem; background: #c67489; color: white; border: none; border-radius: 6px; font-size: 0.95rem; cursor: pointer; font-weight: 600; }
  button:hover { background: #b05568; }
  .msg { font-size: 0.85rem; padding: 0.7rem; border-radius: 6px; margin-bottom: 1rem; display: none; }
  .msg.error { background: #fff0f2; color: #dc2626; border-left: 3px solid #dc2626; display: block; }
  .links { text-align: center; margin-top: 1.5rem; font-size: 0.88rem; }
  .links a { color: #c67489; text-decoration: none; }
</style>
</head>
<body>
<div class="card">
  <h1>Sign In</h1>
  <p>Welcome back to OncoVision AI</p>
  <div id="msg" class="msg"></div>
  <form id="loginForm">
    <input type="text" id="username" placeholder="Username" required/>
    <input type="password" id="password" placeholder="Password" required/>
    <button type="submit">Sign In</button>
  </form>
  <div class="links">
    <a href="/forgot-password">Forgot password?</a><br><br>
    Don't have an account? <a href="/signup">Sign up</a><br>
    <a href="/index.html">Back to Home</a>
  </div>
</div>
<script>
document.getElementById('loginForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const msg = document.getElementById('msg');
  msg.style.display = 'none';
  const resp = await fetch('/login', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    credentials: 'include',
    body: JSON.stringify({
      username: document.getElementById('username').value,
      password: document.getElementById('password').value
    })
  });
  const data = await resp.json();
  if (data.success) {
    window.location.href = data.is_admin ? '/admin' : '/index.html';
  } else {
    msg.className = 'msg error';
    msg.textContent = data.error;
    msg.style.display = 'block';
  }
});
</script>
</body>
</html>''')

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        data  = request.get_json()
        email = data.get('email', '').strip().lower()
        user  = get_user_by_email(email)
        # Always return success — never reveal whether an email exists
        if user and user['email']:
            token      = str(uuid.uuid4())
            expires_at = (datetime.now() + timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
            conn = get_db()
            conn.execute('DELETE FROM password_reset_tokens WHERE user_id=? AND used=0', (user['id'],))
            conn.execute('INSERT INTO password_reset_tokens (user_id, token, expires_at) VALUES (?,?,?)',
                         (user['id'], token, expires_at))
            conn.commit()
            conn.close()
            sent = send_reset_email(user['email'], token)
            if not sent:
                # Email not configured — fall back to showing the link (dev/demo mode)
                return jsonify({'success': True, 'dev_token': token})
        return jsonify({'success': True})

    return render_template_string('''<!DOCTYPE html>
<html>
<head><title>Forgot Password — OncoVision AI</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg"/>
<style>
  body { font-family: sans-serif; background: #fdf8fa; margin: 0; display: flex; align-items: center; justify-content: center; min-height: 100vh; }
  .card { background: white; border-radius: 12px; padding: 2.5rem; max-width: 400px; width: 100%; box-shadow: 0 4px 6px rgba(0,0,0,0.07); }
  h1 { font-size: 1.8rem; margin-bottom: 0.5rem; }
  p  { color: #999; font-size: 0.9rem; margin-bottom: 2rem; }
  input { width: 100%; padding: 0.75rem; margin-bottom: 1rem; border: 1px solid #ddd; border-radius: 6px; font-size: 0.95rem; box-sizing: border-box; }
  input:focus { outline: none; border-color: #c67489; }
  button { width: 100%; padding: 0.85rem; background: #c67489; color: white; border: none; border-radius: 6px; font-size: 0.95rem; cursor: pointer; font-weight: 600; }
  button:hover { background: #b05568; }
  .msg { font-size: 0.85rem; padding: 0.75rem; border-radius: 6px; margin-bottom: 1rem; display: none; }
  .msg.error   { background: #fff0f2; color: #dc2626; border-left: 3px solid #dc2626; display: block; }
  .msg.success { background: #f0fdf4; color: #16a34a; border-left: 3px solid #16a34a; display: block; }
  .dev-box { background: #fffbeb; border: 1px solid #fcd34d; border-radius: 6px; padding: 0.85rem; margin-top: 0.75rem; font-size: 0.78rem; color: #92400e; word-break: break-all; }
  .links { text-align: center; margin-top: 1.5rem; font-size: 0.88rem; }
  .links a { color: #c67489; text-decoration: none; }
</style>
</head>
<body>
<div class="card">
  <h1>Forgot Password</h1>
  <p>Enter your email address and we'll send you a link to reset your password.</p>
  <div id="msg" class="msg"></div>
  <form id="forgotForm">
    <input type="email" id="email" placeholder="Your email address" required/>
    <button type="submit">Send Reset Link</button>
  </form>
  <div class="links"><a href="/login">Back to Sign In</a></div>
</div>
<script>
document.getElementById('forgotForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const msg = document.getElementById('msg');
  msg.className = 'msg'; msg.style.display = 'none';
  const btn = e.target.querySelector('button');
  btn.disabled = true; btn.textContent = 'Sending…';
  const resp = await fetch('/forgot-password', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ email: document.getElementById('email').value.trim() })
  });
  const data = await resp.json();
  document.getElementById('forgotForm').style.display = 'none';
  if (data.dev_token) {
    // Dev/demo fallback — email not configured
    const link = window.location.origin + '/reset-password/' + data.dev_token;
    msg.className = 'msg success';
    msg.innerHTML = 'Email not configured. <strong>Dev mode:</strong> use the link below to reset your password.';
    msg.style.display = 'block';
    const box = document.createElement('div');
    box.className = 'dev-box';
    box.innerHTML = '<a href="' + link + '">' + link + '</a>';
    msg.after(box);
  } else {
    msg.className = 'msg success';
    msg.textContent = 'If that email is registered, a reset link has been sent. Check your inbox.';
    msg.style.display = 'block';
  }
});
</script>
</body>
</html>''')

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    conn = get_db()
    record = conn.execute(
        'SELECT * FROM password_reset_tokens WHERE token=? AND used=0', (token,)
    ).fetchone()

    if not record or record['expires_at'] < datetime.now().strftime('%Y-%m-%d %H:%M:%S'):
        conn.close()
        return render_template_string('''<!DOCTYPE html>
<html><head><title>Link Expired — OncoVision AI</title>
<style>body{font-family:sans-serif;background:#fdf8fa;display:flex;align-items:center;justify-content:center;min-height:100vh;}
.card{background:white;border-radius:12px;padding:2.5rem;max-width:380px;width:100%;box-shadow:0 4px 6px rgba(0,0,0,0.07);text-align:center;}
a{color:#c67489;}</style></head>
<body><div class="card"><h2>Link Expired or Invalid</h2>
<p style="color:#999;">This reset link is invalid or has already been used.</p>
<a href="/forgot-password">Request a new link</a></div></body></html>''')

    if request.method == 'POST':
        data    = request.get_json()
        new_pwd = data.get('password', '')
        errs    = validate_password(new_pwd)
        if errs:
            conn.close()
            return jsonify({'error': 'Password must contain: ' + ', '.join(errs)}), 400
        hashed = bcrypt.hashpw(new_pwd.encode('utf-8'), bcrypt.gensalt())
        conn.execute('UPDATE users SET password=? WHERE id=?',
                     (hashed.decode('utf-8'), record['user_id']))
        conn.execute('UPDATE password_reset_tokens SET used=1 WHERE token=?', (token,))
        conn.commit()
        conn.close()
        return jsonify({'success': True})

    conn.close()
    return render_template_string('''<!DOCTYPE html>
<html>
<head><title>Reset Password — OncoVision AI</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg"/>
<style>
  body { font-family: sans-serif; background: #fdf8fa; margin: 0; display: flex; align-items: center; justify-content: center; min-height: 100vh; }
  .card { background: white; border-radius: 12px; padding: 2.5rem; max-width: 380px; width: 100%; box-shadow: 0 4px 6px rgba(0,0,0,0.07); }
  h1 { font-size: 1.8rem; margin-bottom: 0.5rem; }
  p  { color: #999; font-size: 0.9rem; margin-bottom: 2rem; }
  input { width: 100%; padding: 0.75rem; margin-bottom: 1rem; border: 1px solid #ddd; border-radius: 6px; font-size: 0.95rem; box-sizing: border-box; }
  input:focus { outline: none; border-color: #c67489; }
  button { width: 100%; padding: 0.85rem; background: #c67489; color: white; border: none; border-radius: 6px; font-size: 0.95rem; cursor: pointer; font-weight: 600; }
  button:hover { background: #b05568; }
  .msg { font-size: 0.85rem; padding: 0.7rem; border-radius: 6px; margin-bottom: 1rem; display: none; }
  .msg.error   { background: #fff0f2; color: #dc2626; border-left: 3px solid #dc2626; display: block; }
  .msg.success { background: #f0fdf4; color: #16a34a; border-left: 3px solid #16a34a; display: block; }
</style>
</head>
<body>
<div class="card">
  <h1>Set New Password</h1>
  <p>Choose a strong new password for your account.</p>
  <div id="msg" class="msg"></div>
  <form id="resetForm">
    <input type="password" id="password" placeholder="New password" required/>
    <input type="password" id="confirm"  placeholder="Confirm new password" required/>
    <button type="submit">Reset Password</button>
  </form>
</div>
<script>
const TOKEN = ''' + f'"{token}"' + ''';
document.getElementById('resetForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const msg = document.getElementById('msg');
  msg.className = 'msg'; msg.style.display = 'none';
  const pwd     = document.getElementById('password').value;
  const confirm = document.getElementById('confirm').value;
  if (pwd !== confirm) {
    msg.className = 'msg error'; msg.textContent = 'Passwords do not match.'; msg.style.display = 'block'; return;
  }
  const resp = await fetch('/reset-password/' + TOKEN, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ password: pwd })
  });
  const data = await resp.json();
  if (data.success) {
    msg.className = 'msg success';
    msg.textContent = 'Password reset successfully! Redirecting to sign in…';
    msg.style.display = 'block';
    document.getElementById('resetForm').style.display = 'none';
    setTimeout(() => window.location.href = '/login', 2000);
  } else {
    msg.className = 'msg error'; msg.textContent = data.error; msg.style.display = 'block';
  }
});
</script>
</body>
</html>''')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        data        = request.get_json()
        username    = data.get('username', '').strip()
        password    = data.get('password', '')
        full_name   = data.get('full_name', '').strip()
        role        = data.get('role', '').strip()
        institution = data.get('institution', '').strip()
        email       = data.get('email', '').strip().lower()

        if not email or '@' not in email:
            return jsonify({'error': 'A valid email address is required.'}), 400
        if username_exists(username):
            return jsonify({'error': 'Username already exists'}), 400
        if get_user_by_email(email):
            return jsonify({'error': 'An account with that email already exists.'}), 400

        # Validate password strength
        pwd_errors = validate_password(password)
        if pwd_errors:
            return jsonify({'error': 'Password must contain: ' + ', '.join(pwd_errors) + '.'}), 400

        create_user(username, password, full_name, role, institution, email)
        return jsonify({'success': True})

    return render_template_string('''<!DOCTYPE html>
<html>
<head><title>Sign Up — OncoVision AI</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg"/>
<style>
  body { font-family: sans-serif; background: #fdf8fa; margin: 0; display: flex; align-items: center; justify-content: center; min-height: 100vh; padding: 2rem 0; }
  .card { background: white; border-radius: 12px; padding: 2.5rem; max-width: 420px; width: 100%; box-shadow: 0 4px 6px rgba(0,0,0,0.07); }
  h1 { font-size: 1.8rem; margin-bottom: 0.5rem; }
  p { color: #999; font-size: 0.9rem; margin-bottom: 2rem; }
  input, select { width: 100%; padding: 0.75rem; margin-bottom: 1rem; border: 1px solid #ddd; border-radius: 6px; font-size: 0.95rem; box-sizing: border-box; }
  input:focus, select:focus { outline: none; border-color: #c67489; }
  button { width: 100%; padding: 0.85rem; background: #c67489; color: white; border: none; border-radius: 6px; font-size: 0.95rem; cursor: pointer; font-weight: 600; }
  button:hover { background: #b05568; }
  .msg { font-size: 0.85rem; padding: 0.7rem; border-radius: 6px; margin-bottom: 1rem; display: none; }
  .msg.error { background: #fff0f2; color: #dc2626; border-left: 3px solid #dc2626; }
  .msg.success { background: #f0fdf4; color: #16a34a; border-left: 3px solid #16a34a; }
  .links { text-align: center; margin-top: 1.5rem; font-size: 0.88rem; }
  .links a { color: #c67489; text-decoration: none; }
  .pwd-hint { font-size: 0.78rem; color: #999; margin-top: -0.6rem; margin-bottom: 1rem; }
  .terms-row { display: flex; align-items: flex-start; gap: 0.5rem; margin-bottom: 1rem; font-size: 0.82rem; color: #666; }
  .terms-row input[type="checkbox"] { width: auto; margin: 0; margin-top: 2px; flex-shrink: 0; }
  .terms-row a { color: #c67489; text-decoration: none; }
  .disclaimer-box { background: #fff8f0; border-left: 3px solid #f59e0b; border-radius: 6px; padding: 0.65rem 0.9rem; font-size: 0.78rem; color: #92400e; margin-bottom: 1.25rem; line-height: 1.4; }
</style>
</head>
<body>
<div class="card">
  <h1>Create Account</h1>
  <p>Join OncoVision AI to save and track your predictions</p>
  <div class="disclaimer-box">
    ⚕️ <strong>Medical Disclaimer:</strong> OncoVision AI is a research prototype.
    It is <strong>not</strong> a validated diagnostic tool and must not replace professional clinical judgement.
  </div>
  <div id="msg" class="msg"></div>
  <form id="signupForm">
    <input type="text"  id="username"    placeholder="Username" required/>
    <input type="text"  id="fullName"    placeholder="Full Name" required/>
    <input type="email" id="email"       placeholder="Email address" required/>
    <select id="role" required>
      <option value="">Select your role</option>
      <option value="Pathologist">Pathologist</option>
      <option value="Clinician">Clinician</option>
      <option value="Radiologist">Radiologist</option>
      <option value="Researcher">Researcher</option>
      <option value="Other">Other</option>
    </select>
    <input type="text"     id="institution" placeholder="Institution (optional)"/>
    <input type="password" id="password"    placeholder="Password" required/>
    <div class="pwd-hint">Min 8 chars, uppercase, number, special character</div>
    <div class="terms-row">
      <input type="checkbox" id="termsCheck" required/>
      <label for="termsCheck">
        I understand that OncoVision AI is a research tool and
        <strong>not</strong> a substitute for professional medical diagnosis.
        I agree to use it responsibly.
      </label>
    </div>
    <button type="submit">Create Account</button>
  </form>
  <div class="links">
    Already have an account? <a href="/login">Sign in</a>
  </div>
</div>
<script>
document.getElementById('signupForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const msg = document.getElementById('msg');
  msg.style.display = 'none';
  const resp = await fetch('/signup', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      username:    document.getElementById('username').value,
      password:    document.getElementById('password').value,
      full_name:   document.getElementById('fullName').value,
      email:       document.getElementById('email').value,
      role:        document.getElementById('role').value,
      institution: document.getElementById('institution').value
    })
  });
  const data = await resp.json();
  if (data.success) {
    msg.className = 'msg success';
    msg.textContent = 'Account created! Redirecting to sign in...';
    msg.style.display = 'block';
    setTimeout(() => window.location.href = '/login', 1500);
  } else {
    msg.className = 'msg error';
    msg.textContent = data.error;
    msg.style.display = 'block';
  }
});
</script>
</body>
</html>''')

@app.route('/logout')
def logout():
    log_audit('LOGOUT')
    session.clear()
    response = redirect('/login')
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Clear-Site-Data'] = '"cache", "storage"'
    return response

# ── ADMIN DASHBOARD ───────────────────────────────────────────────────
@app.route('/admin')
@login_required
@admin_required
def admin_dashboard():
    users = get_all_users()
    preds = get_all_predictions()
    stats = get_stats()
    
    rows_users = ''.join([
        f'''<tr>
          <td style="font-weight:500;color:#333;">{u['full_name']}</td>
          <td>{u['role']}</td>
          <td>{u['institution'] or '—'}</td>
          <td style="font-family:monospace;font-size:0.82rem;">{u['username']}</td>
          <td style="text-align:center;">{u['prediction_count']}</td>
          <td style="font-size:0.82rem;color:#999;">{u['created_at']}</td>
          <td style="white-space:nowrap;">
            <button class="act-btn act-btn--promote" onclick="promoteUser({u['id']}, '{u['username']}')">Make Admin</button>
            <button class="act-btn act-btn--delete" onclick="deleteUser({u['id']}, '{u['username']}')">Delete</button>
          </td>
        </tr>''' for u in users
    ])
    
    rows_preds = ''.join([
        f'''<tr>
          <td style="font-weight:500;color:#333;">{p['patient_name']}</td>
          <td><span class="badge-{'mal' if p['prediction']=='Malignant' else 'ben'}">{p['prediction']}</span></td>
          <td>{round(p['confidence']*100,1)}%</td>
          <td>{p['full_name'] or p['username']}</td>
          <td style="font-size:0.82rem;color:#999;">{p['created_at']}</td>
          <td>
            <button class="act-btn act-btn--delete" onclick="deletePrediction('{p['prediction_id']}')">Delete</button>
          </td>
        </tr>''' for p in preds
    ])
    
    mal_pct = stats['mal_pct']
    
    return render_template_string(f'''<!DOCTYPE html>
<html>
<head>
  <title>Admin Dashboard — OncoVision AI</title>
  <link rel="icon" type="image/svg+xml" href="/favicon.svg"/>
  <link rel="stylesheet" href="/style.css"/>
  <script src="/darkmode.js"></script>
  <style>
    .container {{ max-width:1400px; margin:0 auto; padding:2.5rem 2rem; }}
    .stats-grid {{ display:grid; grid-template-columns:repeat(5,1fr); gap:1.5rem; margin-bottom:2.5rem; }}
    .stat-card {{ background:var(--admin-card,white); border-radius:10px; padding:1.5rem; border:1px solid var(--admin-border,#eee); }}
    .stat-card__num {{ font-size:2rem; font-weight:600; margin-bottom:0.3rem; color:var(--text,#111); }}
    .stat-card__label {{ font-size:0.8rem; color:var(--text-muted,#999); text-transform:uppercase; letter-spacing:0.06em; }}
    .section {{ background:var(--admin-card,white); border-radius:10px; border:1px solid var(--admin-border,#eee); margin-bottom:2rem; overflow:hidden; }}
    .section__header {{ padding:1.2rem 1.5rem; border-bottom:1px solid var(--admin-border,#eee); display:flex; justify-content:space-between; align-items:center; }}
    .section__header h2 {{ font-size:1rem; font-weight:600; color:var(--text,#111); }}
    .section__header span {{ font-size:0.82rem; color:var(--text-muted,#999); }}
    table {{ width:100%; border-collapse:collapse; font-size:0.88rem; }}
    th {{ text-align:left; padding:0.85rem 1.2rem; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; color:var(--text-muted,#999); border-bottom:1px solid var(--admin-border,#eee); font-weight:600; }}
    td {{ padding:0.85rem 1.2rem; border-bottom:1px solid var(--admin-border,#f5f5f5); vertical-align:middle; color:var(--text,#111); }}
    tr:last-child td {{ border-bottom:none; }}
    tr:hover td {{ background:var(--admin-hover,#fdf8fa); }}
    .badge-mal {{ background:#fee2e2; color:#dc2626; padding:0.2rem 0.6rem; border-radius:100px; font-size:0.78rem; font-weight:600; }}
    .badge-ben {{ background:#dcfce7; color:#16a34a; padding:0.2rem 0.6rem; border-radius:100px; font-size:0.78rem; font-weight:600; }}
    .bar-wrap {{ background:#f0f0f0; border-radius:6px; height:6px; overflow:hidden; margin-top:0.5rem; }}
    .bar-mal {{ height:100%; background:#dc2626; border-radius:6px; width:{mal_pct}%; }}
    .empty {{ text-align:center; padding:3rem; color:#999; font-size:0.9rem; }}
    .act-btn {{ border:none; border-radius:6px; padding:0.3rem 0.75rem; font-size:0.78rem; font-weight:600; cursor:pointer; transition:opacity 0.15s; }}
    .act-btn:hover {{ opacity:0.8; }}
    .act-btn--promote {{ background:#dbeafe; color:#1d4ed8; margin-right:0.3rem; }}
    .act-btn--delete {{ background:#fee2e2; color:#dc2626; }}
    .modal-overlay {{ display:none; position:fixed; inset:0; background:rgba(0,0,0,0.45); z-index:1000; align-items:center; justify-content:center; }}
    .modal-overlay.open {{ display:flex; }}
    .modal {{ background:var(--admin-card,white); border-radius:12px; padding:2rem; width:100%; max-width:440px; box-shadow:0 20px 60px rgba(0,0,0,0.2); }}
    .modal h2 {{ font-size:1.1rem; font-weight:700; margin-bottom:1.25rem; color:var(--text,#111); }}
    .modal label {{ display:block; font-size:0.8rem; color:var(--text-muted,#666); margin-bottom:0.25rem; margin-top:0.75rem; font-weight:600; text-transform:uppercase; letter-spacing:0.04em; }}
    .modal input, .modal select {{ width:100%; padding:0.6rem 0.85rem; border:1px solid var(--admin-border,#ddd); border-radius:7px; font-size:0.9rem; outline:none; background:var(--admin-field,white); color:var(--text,#111); }}
    .modal input:focus, .modal select:focus {{ border-color:#c67489; }}
    .modal-actions {{ display:flex; gap:0.75rem; margin-top:1.5rem; justify-content:flex-end; }}
    .btn-primary {{ background:#c67489; color:white; border:none; border-radius:7px; padding:0.6rem 1.25rem; font-size:0.9rem; font-weight:600; cursor:pointer; }}
    .btn-cancel {{ background:var(--admin-btn-cancel,#f3f4f6); color:var(--text,#374151); border:none; border-radius:7px; padding:0.6rem 1.25rem; font-size:0.9rem; font-weight:600; cursor:pointer; }}
    .msg {{ padding:0.6rem 1rem; border-radius:7px; font-size:0.85rem; margin-bottom:1rem; display:none; }}
    .msg--err {{ background:#fee2e2; color:#dc2626; }}
    .msg--ok {{ background:#dcfce7; color:#16a34a; }}
    .pagination {{ display:flex; align-items:center; gap:0.4rem; padding:1rem 1.5rem; border-top:1px solid var(--admin-border,#eee); justify-content:flex-end; flex-wrap:wrap; }}
    .pg-btn {{ background:var(--admin-card,white); border:1px solid var(--admin-border,#ddd); border-radius:6px; padding:0.35rem 0.7rem; font-size:0.82rem; cursor:pointer; color:var(--text,#333); }}
    .pg-btn:hover {{ border-color:#c67489; color:#c67489; }}
    .pg-btn.active {{ background:#c67489; color:white; border-color:#c67489; }}
    .pg-btn:disabled {{ opacity:0.4; cursor:default; }}
    .pg-info {{ font-size:0.82rem; color:var(--text-muted,#999); margin-right:auto; }}
    @media(max-width:768px) {{ .stats-grid {{ grid-template-columns:repeat(2,1fr); }} }}
    [data-theme="dark"] {{
      --admin-card:       #1a0e13;
      --admin-border:     #2e1a22;
      --admin-hover:      #2a0f18;
      --admin-field:      #0f0a0d;
      --admin-btn-cancel: #2a0f18;
    }}
    [data-theme="dark"] .modal select option {{ background:#1a0e13; color:var(--text); }}
  </style>
</head>
<body>
  <nav class="nav">
    <div class="nav__inner">
      <a href="/index.html" class="nav__logo">OncoVision AI</a>
      <ul class="nav__links">
        <li><a href="/index.html">Main Site</a></li>
        <li><a href="/analytics.html">Analytics</a></li>
        <li><a href="/audit.html">Audit Log</a></li>
        <li><a href="/admin" class="active">Admin Dashboard</a></li>
        <li><span style="font-size:0.82rem;color:var(--text-muted);">{session.get('full_name','Admin')}</span></li>
        <li><a href="/logout">Sign out</a></li>
      </ul>
    </div>
  </nav>
  <div class="container">
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-card__num">{stats['total_users']}</div>
        <div class="stat-card__label">Registered Users</div>
      </div>
      <div class="stat-card">
        <div class="stat-card__num">{stats['total_preds']}</div>
        <div class="stat-card__label">Total Predictions</div>
      </div>
      <div class="stat-card">
        <div class="stat-card__num" style="color:#dc2626;">{stats['malignant']}</div>
        <div class="stat-card__label">Malignant</div>
        <div class="bar-wrap"><div class="bar-mal"></div></div>
        <div style="font-size:0.75rem;color:#999;margin-top:0.3rem;">{mal_pct}% of total</div>
      </div>
      <div class="stat-card">
        <div class="stat-card__num" style="color:#16a34a;">{stats['benign']}</div>
        <div class="stat-card__label">Benign</div>
      </div>
      <div class="stat-card">
        <div class="stat-card__num">{stats['today_preds']}</div>
        <div class="stat-card__label">Predictions Today</div>
      </div>
    </div>
    <div class="section">
      <div class="section__header">
        <h2>Registered Users</h2>
        <div style="display:flex;align-items:center;gap:1rem;">
          <span>{stats['total_users']} total</span>
          <button class="act-btn act-btn--promote" onclick="openCreateModal()" style="background:#c67489;color:white;">+ Create User</button>
        </div>
      </div>
      {'<table><thead><tr><th>Full Name</th><th>Role</th><th>Institution</th><th>Username</th><th style="text-align:center;">Predictions</th><th>Joined</th><th>Actions</th></tr></thead><tbody>' + rows_users + '</tbody></table>' if users else '<div class="empty">No users registered yet.</div>'}
    </div>
    <div class="section">
      <div class="section__header">
        <h2>Recent Predictions</h2>
        <span id="predCount">{len(preds)} records</span>
      </div>
      {'<table><thead><tr><th>Patient Name</th><th>Result</th><th>Confidence</th><th>Analysed By</th><th>Date and Time</th><th>Actions</th></tr></thead><tbody id="predsTbody">' + rows_preds + '</tbody></table><div class="pagination" id="predPagination"></div>' if preds else '<div class="empty">No predictions recorded yet.</div>'}
    </div>
  </div>

  <!-- CREATE USER MODAL -->
  <div class="modal-overlay" id="createModal">
    <div class="modal">
      <h2>Create New User</h2>
      <div class="msg msg--err" id="createErr"></div>
      <div class="msg msg--ok" id="createOk"></div>
      <label>Full Name</label>
      <input type="text" id="cu_name" placeholder="Dr. Jane Doe"/>
      <label>Username</label>
      <input type="text" id="cu_username" placeholder="jane.doe"/>
      <label>Password</label>
      <input type="password" id="cu_password" placeholder="Min 8 chars, 1 uppercase, 1 number, 1 special"/>
      <label>Role</label>
      <select id="cu_role">
        <option value="">Select role</option>
        <option>Radiologist</option>
        <option>Pathologist</option>
        <option>Oncologist</option>
        <option>General Physician</option>
        <option>Researcher</option>
        <option>Administrator</option>
      </select>
      <label>Institution</label>
      <input type="text" id="cu_institution" placeholder="Hospital / University"/>
      <label style="display:flex;align-items:center;gap:0.5rem;margin-top:0.9rem;">
        <input type="checkbox" id="cu_is_admin" style="width:auto;"/> Grant Admin Access
      </label>
      <div class="modal-actions">
        <button class="btn-cancel" onclick="closeCreateModal()">Cancel</button>
        <button class="btn-primary" onclick="submitCreateUser()">Create User</button>
      </div>
    </div>
  </div>

  <script>
    const API = '';  // same origin

    function deleteUser(userId, username) {{
      if (!confirm('Delete user "' + username + '" and all their predictions? This cannot be undone.')) return;
      fetch(API + '/api/admin/users/' + userId, {{method:'DELETE', credentials:'include'}})
        .then(r => r.json())
        .then(d => {{
          if (d.success) location.reload();
          else alert('Error: ' + d.error);
        }});
    }}

    function promoteUser(userId, username) {{
      if (!confirm('Grant admin access to "' + username + '"?')) return;
      fetch(API + '/api/admin/users/' + userId + '/toggle-admin', {{method:'POST', credentials:'include'}})
        .then(r => r.json())
        .then(d => {{
          if (d.success) location.reload();
          else alert('Error: ' + d.error);
        }});
    }}

    function deletePrediction(predId) {{
      if (!confirm('Delete this prediction record? This cannot be undone.')) return;
      fetch(API + '/api/admin/predictions/' + predId, {{method:'DELETE', credentials:'include'}})
        .then(r => r.json())
        .then(d => {{
          if (d.success) location.reload();
          else alert('Error: ' + d.error);
        }});
    }}

    function openCreateModal() {{
      document.getElementById('createModal').classList.add('open');
      document.getElementById('createErr').style.display = 'none';
      document.getElementById('createOk').style.display = 'none';
    }}

    function closeCreateModal() {{
      document.getElementById('createModal').classList.remove('open');
    }}

    async function submitCreateUser() {{
      const err = document.getElementById('createErr');
      const ok  = document.getElementById('createOk');
      err.style.display = 'none'; ok.style.display = 'none';

      const body = {{
        full_name:   document.getElementById('cu_name').value.trim(),
        username:    document.getElementById('cu_username').value.trim(),
        password:    document.getElementById('cu_password').value,
        role:        document.getElementById('cu_role').value,
        institution: document.getElementById('cu_institution').value.trim(),
        is_admin:    document.getElementById('cu_is_admin').checked
      }};

      const r = await fetch(API + '/api/admin/users/create', {{
        method: 'POST',
        credentials: 'include',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify(body)
      }});
      const d = await r.json();
      if (d.success) {{
        ok.textContent = 'User created successfully!';
        ok.style.display = 'block';
        setTimeout(() => {{ closeCreateModal(); location.reload(); }}, 1000);
      }} else {{
        err.textContent = d.error || 'Something went wrong.';
        err.style.display = 'block';
      }}
    }}

    // Close modal on overlay click
    document.getElementById('createModal').addEventListener('click', function(e) {{
      if (e.target === this) closeCreateModal();
    }});

    // ── PREDICTIONS PAGINATION ────────────────────────────────────────
    (function() {{
      const tbody = document.getElementById('predsTbody');
      const pgDiv = document.getElementById('predPagination');
      if (!tbody || !pgDiv) return;

      const rows   = Array.from(tbody.querySelectorAll('tr'));
      const PER    = 20;
      let   page   = 1;
      const total  = rows.length;
      const pages  = Math.ceil(total / PER);

      function render() {{
        rows.forEach((r, i) => {{
          r.style.display = (i >= (page-1)*PER && i < page*PER) ? '' : 'none';
        }});
        pgDiv.innerHTML = '';

        const info = document.createElement('span');
        info.className = 'pg-info';
        const from = (page-1)*PER + 1, to = Math.min(page*PER, total);
        info.textContent = 'Showing ' + from + '–' + to + ' of ' + total;
        pgDiv.appendChild(info);

        const prev = document.createElement('button');
        prev.className = 'pg-btn'; prev.textContent = '← Prev';
        prev.disabled = page === 1;
        prev.onclick = () => {{ page--; render(); }};
        pgDiv.appendChild(prev);

        const maxBtns = 7;
        let start = Math.max(1, page - Math.floor(maxBtns/2));
        let end   = Math.min(pages, start + maxBtns - 1);
        if (end - start < maxBtns - 1) start = Math.max(1, end - maxBtns + 1);

        for (let p = start; p <= end; p++) {{
          const btn = document.createElement('button');
          btn.className = 'pg-btn' + (p === page ? ' active' : '');
          btn.textContent = p;
          btn.onclick = ((_p) => () => {{ page = _p; render(); }})(p);
          pgDiv.appendChild(btn);
        }}

        const next = document.createElement('button');
        next.className = 'pg-btn'; next.textContent = 'Next →';
        next.disabled = page === pages;
        next.onclick = () => {{ page++; render(); }};
        pgDiv.appendChild(next);
      }}

      if (pages > 1) render();
    }})();
  </script>
</body>
</html>''')

# ── PROFILE ROUTES ────────────────────────────────────────────────────
@app.route('/api/profile', methods=['GET'])
@login_required
def get_profile():
    user = get_user(session.get('username'))
    if not user:
        return jsonify({'error': 'User not found'}), 404
    return jsonify({
        'username':    user['username'],
        'full_name':   user['full_name'],
        'email':       user['email'] or '',
        'role':        user['role'],
        'institution': user['institution'] or '',
        'created_at':  user['created_at'],
    })

@app.route('/api/profile/update', methods=['POST'])
@login_required
def update_profile():
    data        = request.get_json()
    full_name   = data.get('full_name', '').strip()
    role        = data.get('role', '').strip()
    institution = data.get('institution', '').strip()
    email       = data.get('email', '').strip().lower()

    if not full_name:
        return jsonify({'error': 'Full name is required.'}), 400
    if not role:
        return jsonify({'error': 'Role is required.'}), 400
    if email and '@' not in email:
        return jsonify({'error': 'Please enter a valid email address.'}), 400

    # Check email not taken by another user
    if email:
        existing = get_user_by_email(email)
        if existing and existing['id'] != session.get('user_id'):
            return jsonify({'error': 'That email is already in use by another account.'}), 400

    user_id = session.get('user_id')
    update_user(user_id, full_name, role, institution, email)
    session['full_name'] = full_name
    session['role']      = role
    return jsonify({'success': True, 'full_name': full_name, 'role': role})

@app.route('/api/profile/change-password', methods=['POST'])
@login_required
def change_password():
    data         = request.get_json()
    current_pwd  = data.get('current_password', '')
    new_pwd      = data.get('new_password', '')
    confirm_pwd  = data.get('confirm_password', '')

    username = session.get('username')
    if not check_password(username, current_pwd):
        return jsonify({'error': 'Current password is incorrect.'}), 400

    pwd_errors = validate_password(new_pwd)
    if pwd_errors:
        return jsonify({'error': 'New password must contain: ' + ', '.join(pwd_errors) + '.'}), 400

    if new_pwd != confirm_pwd:
        return jsonify({'error': 'New passwords do not match.'}), 400

    update_password(session.get('user_id'), new_pwd)
    return jsonify({'success': True})

@app.route('/api/account/delete', methods=['POST'])
@login_required
def delete_account():
    data     = request.get_json()
    password = data.get('password', '')
    username = session.get('username')

    if not check_password(username, password):
        return jsonify({'error': 'Incorrect password.'}), 401

    user_id = session.get('user_id')
    conn = get_db()
    conn.execute('DELETE FROM predictions WHERE user_id=?', (user_id,))
    conn.execute('DELETE FROM password_reset_tokens WHERE user_id=?', (user_id,))
    conn.execute('DELETE FROM users WHERE id=?', (user_id,))
    conn.commit()
    conn.close()

    session.clear()
    return jsonify({'success': True})

# ── API ROUTES ────────────────────────────────────────────────────────
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status':     'ok',
        'model':      MODEL_PATH,
        'shap_ready': shap_explainer is not None,
        'threshold':  THRESHOLD
    })

@app.route('/api/check-auth', methods=['GET'])
def check_auth():
    return jsonify({
        'logged_in': session.get('logged_in', False),
        'username':  session.get('username', None),
        'full_name': session.get('full_name', None),
        'role':      session.get('role', None),
        'is_admin':  session.get('is_admin', False)
    })

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    
    # SECURITY: Validate file
    if not file.filename:
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    file_bytes = file.read()
    file.seek(0)  # Reset file pointer
    
    if not validate_file_size(file_bytes):
        return jsonify({'error': f'File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.0f}MB'}), 400
    
    # SECURITY: Sanitize filename
    filename = secure_filename(file.filename)
    patient_name = request.form.get('patient_name', '').strip()
    do_gradcam   = request.form.get('gradcam', 'true').lower() == 'true'
    do_shap      = request.form.get('shap', 'true').lower() == 'true'
    is_logged_in = session.get('logged_in', False)
    user_id      = session.get('user_id', None)
    username     = session.get('username', None)

    try:
        raw_img   = load_image(file_bytes)
        img_input = to_model_input(raw_img)

        prob       = float(model.predict(img_input, verbose=0)[0][0])
        prediction = 'Malignant' if prob > THRESHOLD else 'Benign'
        confidence = prob if prediction == 'Malignant' else 1 - prob

        gradcam_result = None
        shap_result = None
        original_img_filename = None
        gradcam_img_filename = None
        shap_img_filename = None

        if do_gradcam:
            gradcam_result = generate_gradcam(img_input, raw_img)
        
        if do_shap and get_shap_explainer() is not None:
            shap_result = generate_shap(img_input, raw_img)
        
        # Save images permanently if logged in
        if is_logged_in and not session.get('is_admin'):
            if gradcam_result:
                original_img_filename = save_image_base64(gradcam_result['original'], 'original')
                gradcam_img_filename = save_image_base64(gradcam_result['overlay'], 'gradcam')
            
            if shap_result:
                shap_img_filename = save_image_base64(shap_result['overlay'], 'shap')
            
            log_prediction(user_id, username, patient_name, filename, prediction, confidence, prob,
                          original_img_filename, gradcam_img_filename, shap_img_filename)
            log_audit('PREDICTION', f'{prediction} ({confidence:.1%}) — file: {filename}')

        return jsonify({
            'prediction':   prediction,
            'confidence':   round(confidence, 4),
            'probability':  round(prob, 4),
            'patient_name': patient_name or 'Unknown',
            'saved':        is_logged_in and not session.get('is_admin'),
            'gradcam':      gradcam_result,
            'shap':         shap_result,
        })

    except Exception as e:
        import traceback
        print(f'Error during analysis: {e}')
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/logs', methods=['GET'])
@login_required
def get_logs():
    user_id = session.get('user_id')
    logs    = get_user_logs(user_id)
    return jsonify({'logs': logs, 'total': len(logs)})

# ── PREDICTION DETAIL API ─────────────────────────────────────────────
@app.route('/api/prediction/<prediction_id>', methods=['GET'])
@login_required
def get_prediction_detail(prediction_id):
    user_id = session.get('user_id')
    prediction = get_prediction_by_id(prediction_id, user_id)
    
    if not prediction:
        return jsonify({'error': 'Prediction not found'}), 404
    
    return jsonify(prediction)

# ── PREDICTION MANAGEMENT ROUTES ─────────────────────────────────────
@app.route('/api/prediction/<prediction_id>/update', methods=['POST'])
@login_required
def update_prediction(prediction_id):
    user_id = session.get('user_id')
    data = request.get_json()
    patient_name = data.get('patient_name', '').strip()
    notes = data.get('notes', '').strip()

    conn = get_db()
    # Ensure notes column exists
    try:
        conn.execute('ALTER TABLE predictions ADD COLUMN notes TEXT')
        conn.commit()
    except sqlite3.OperationalError:
        pass

    result = conn.execute(
        'UPDATE predictions SET patient_name=?, notes=? WHERE prediction_id=? AND user_id=?',
        (patient_name, notes, prediction_id, user_id)
    )
    conn.commit()
    conn.close()

    if result.rowcount == 0:
        return jsonify({'error': 'Prediction not found'}), 404
    return jsonify({'success': True})

@app.route('/api/prediction/<prediction_id>/delete', methods=['DELETE'])
@login_required
def delete_prediction(prediction_id):
    user_id = session.get('user_id')
    conn = get_db()
    row = conn.execute(
        'SELECT * FROM predictions WHERE prediction_id=? AND user_id=?',
        (prediction_id, user_id)
    ).fetchone()

    if not row:
        conn.close()
        return jsonify({'error': 'Prediction not found'}), 404

    # Delete associated image files
    for col in ['original_image', 'gradcam_image', 'shap_image']:
        fname = row[col]
        if fname:
            try:
                os.remove(os.path.join(HISTORY_FOLDER, fname))
            except FileNotFoundError:
                pass

    conn.execute(
        'DELETE FROM predictions WHERE prediction_id=? AND user_id=?',
        (prediction_id, user_id)
    )
    conn.commit()
    conn.close()
    return jsonify({'success': True})

@app.route('/api/predictions/bulk-delete', methods=['POST'])
@login_required
def bulk_delete_predictions():
    user_id = session.get('user_id')
    data = request.get_json()
    ids = data.get('prediction_ids', [])

    if not ids:
        return jsonify({'error': 'No IDs provided'}), 400

    conn = get_db()
    deleted = 0
    for pid in ids:
        row = conn.execute(
            'SELECT * FROM predictions WHERE prediction_id=? AND user_id=?',
            (pid, user_id)
        ).fetchone()
        if row:
            for col in ['original_image', 'gradcam_image', 'shap_image']:
                fname = row[col]
                if fname:
                    try:
                        os.remove(os.path.join(HISTORY_FOLDER, fname))
                    except FileNotFoundError:
                        pass
            conn.execute(
                'DELETE FROM predictions WHERE prediction_id=? AND user_id=?',
                (pid, user_id)
            )
            deleted += 1

    conn.commit()
    conn.close()
    return jsonify({'success': True, 'deleted': deleted})

# ── SERVE HISTORY IMAGES ──────────────────────────────────────────────
@app.route('/history_images/<filename>')
@login_required
def serve_history_image(filename):
    # SECURITY: Validate filename to prevent directory traversal
    safe_filename = secure_filename(filename)
    return send_from_directory(HISTORY_FOLDER, safe_filename)

# ── RUN ───────────────────────────────────────────────────────────────
# Serve HTML files
@app.route('/')
def index():
    return send_file('index.html')

@app.route('/index.html')
def index_html():
    return send_file('index.html')

@app.route('/history.html')
def history_html():
    if not session.get('logged_in'):
        return redirect('/login')
    resp = make_response(send_file('history.html'))
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    return resp

@app.route('/profile.html')
def profile_html():
    if not session.get('logged_in'):
        return redirect('/login')
    resp = make_response(send_file('profile.html'))
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    return resp

@app.route('/analyze.html')
def analyze_html():
    return send_file('analyze.html')

@app.route('/about.html')
def about_html():
    return send_file('about.html')

@app.route('/style.css')
def style_css():
    return send_file('style.css')

@app.route('/darkmode.js')
def darkmode_js():
    return send_file('darkmode.js')

@app.route('/analytics.html')
def analytics_html():
    if not session.get('logged_in'):
        return redirect('/login')
    resp = make_response(send_file('analytics.html'))
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    return resp

@app.route('/audit.html')
def audit_html():
    return send_file('audit.html')

# ── ANALYTICS API ─────────────────────────────────────────────────────
@app.route('/api/analytics')
@login_required
def analytics():
    days     = request.args.get('days', 30, type=int)
    is_admin = session.get('is_admin', False)
    user_id  = session.get('user_id')

    cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')
    conn   = get_db()

    if is_admin:
        rows = conn.execute('''
            SELECT p.prediction, p.confidence, p.created_at,
                   p.filename, u.role,
                   COALESCE(p.username, 'unknown') as username
            FROM predictions p
            LEFT JOIN users u ON p.user_id = u.id
            WHERE p.created_at >= ?
            ORDER BY p.created_at ASC
        ''', (cutoff,)).fetchall()
    else:
        rows = conn.execute('''
            SELECT p.prediction, p.confidence, p.created_at,
                   p.filename, u.role,
                   COALESCE(p.username, 'unknown') as username
            FROM predictions p
            LEFT JOIN users u ON p.user_id = u.id
            WHERE p.user_id = ? AND p.created_at >= ?
            ORDER BY p.created_at ASC
        ''', (user_id, cutoff)).fetchall()

    conn.close()

    preds      = [dict(r) for r in rows]
    total      = len(preds)
    malignant  = sum(1 for p in preds if p['prediction'] == 'Malignant')
    benign     = total - malignant
    avg_conf   = (sum(p['confidence'] for p in preds) / total) if total else 0

    # Normalise date field to YYYY-MM-DD for the frontend
    for p in preds:
        p['date'] = p['created_at'][:10]

    return jsonify({
        'is_admin': bool(is_admin),
        'predictions': preds,
        'summary': {
            'total':          total,
            'malignant':      malignant,
            'benign':         benign,
            'avg_confidence': round(avg_conf, 4),
        }
    })

# ── AUDIT LOG ROUTES ──────────────────────────────────────────────────
@app.route('/api/audit-logs')
@login_required
@admin_required
def get_audit_logs_route():
    limit  = request.args.get('limit', 200, type=int)
    action = request.args.get('action', None)
    user_f = request.args.get('username', None)

    conn = get_db()
    query  = 'SELECT * FROM audit_logs WHERE 1=1'
    params = []

    if action:
        query  += ' AND action = ?'
        params.append(action)
    if user_f:
        query  += ' AND username LIKE ?'
        params.append(f'%{user_f}%')

    query += ' ORDER BY created_at DESC LIMIT ?'
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return jsonify({'logs': [dict(r) for r in rows], 'total': len(rows)})

@app.route('/api/audit-logs/export')
@login_required
@admin_required
def export_audit_logs():
    """Export audit logs as CSV download."""
    import csv, io as _io
    conn = get_db()
    rows = conn.execute('SELECT * FROM audit_logs ORDER BY created_at DESC').fetchall()
    conn.close()

    output = _io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['id','username','action','detail','ip_address','created_at'])
    for r in rows:
        writer.writerow([r['id'], r['username'], r['action'],
                         r['detail'], r['ip_address'], r['created_at']])

    output.seek(0)
    return send_file(
        _io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'oncovision-audit-{datetime.now().strftime("%Y%m%d")}.csv'
    )

# ── ADMIN CRUD ROUTES ─────────────────────────────────────────────────
@app.route('/api/admin/users/create', methods=['POST'])
@login_required
@admin_required
def admin_create_user():
    data        = request.get_json()
    username    = data.get('username', '').strip()
    password    = data.get('password', '').strip()
    full_name   = data.get('full_name', '').strip()
    role        = data.get('role', '').strip()
    institution = data.get('institution', '').strip()
    make_admin  = int(bool(data.get('is_admin', False)))

    if not all([username, password, full_name, role]):
        return jsonify({'error': 'Username, password, full name, and role are required.'}), 400
    if username_exists(username):
        return jsonify({'error': 'Username already taken.'}), 409
    errs = validate_password(password)
    if errs:
        return jsonify({'error': 'Password must contain: ' + ', '.join(errs)}), 400

    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    conn = get_db()
    conn.execute('''
        INSERT INTO users (username, password, full_name, role, institution, is_admin, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (username, hashed.decode('utf-8'), full_name, role, institution, make_admin,
          datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()
    conn.close()
    log_audit('CREATE_USER', f'Created user: {username}' + (' (admin)' if make_admin else ''))
    return jsonify({'success': True})

@app.route('/api/admin/users/<int:user_id>', methods=['DELETE'])
@login_required
@admin_required
def admin_delete_user(user_id):
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id=? AND is_admin=0', (user_id,)).fetchone()
    if not user:
        conn.close()
        return jsonify({'error': 'User not found or cannot delete an admin account.'}), 404
    conn.execute('DELETE FROM predictions WHERE user_id=?', (user_id,))
    conn.execute('DELETE FROM users WHERE id=?', (user_id,))
    conn.commit()
    conn.close()
    log_audit('DELETE_USER', f'Deleted user: {user["username"]}')
    return jsonify({'success': True})

@app.route('/api/admin/users/<int:user_id>/toggle-admin', methods=['POST'])
@login_required
@admin_required
def admin_toggle_admin(user_id):
    if user_id == session.get('user_id'):
        return jsonify({'error': 'Cannot change your own admin status.'}), 400
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id=?', (user_id,)).fetchone()
    if not user:
        conn.close()
        return jsonify({'error': 'User not found.'}), 404
    new_status = 0 if user['is_admin'] else 1
    conn.execute('UPDATE users SET is_admin=? WHERE id=?', (new_status, user_id))
    conn.commit()
    conn.close()
    action = 'PROMOTE_ADMIN' if new_status else 'DEMOTE_ADMIN'
    log_audit(action, f'User: {user["username"]}')
    return jsonify({'success': True, 'is_admin': bool(new_status)})

@app.route('/api/admin/predictions/<prediction_id>', methods=['DELETE'])
@login_required
@admin_required
def admin_delete_prediction(prediction_id):
    conn = get_db()
    pred = conn.execute('SELECT * FROM predictions WHERE prediction_id=?', (prediction_id,)).fetchone()
    if not pred:
        conn.close()
        return jsonify({'error': 'Prediction not found.'}), 404
    conn.execute('DELETE FROM predictions WHERE prediction_id=?', (prediction_id,))
    conn.commit()
    conn.close()
    log_audit('DELETE_PREDICTION', f'Deleted prediction: {prediction_id}')
    return jsonify({'success': True})

# ── PDF EXPORT ────────────────────────────────────────────────────────
@app.route('/api/prediction/<prediction_id>/pdf')
@login_required
def export_prediction_pdf(prediction_id):
    """Generate a professional PDF report for a single prediction."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Image as RLImage, Table, TableStyle,
                                        HRFlowable)
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    except ImportError:
        return jsonify({'error': 'reportlab not installed. Run: pip install reportlab'}), 500

    user_id = session.get('user_id')
    pred    = get_prediction_by_id(prediction_id, user_id)
    if not pred:
        return jsonify({'error': 'Prediction not found'}), 404

    log_audit('PDF_EXPORT', f'Exported PDF for prediction {prediction_id}')

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm,   bottomMargin=2*cm,
        title='OncoVision AI Prediction Report'
    )

    W, H = A4
    styles = getSampleStyleSheet()

    # ── Custom styles
    PINK   = colors.HexColor('#e8326a')
    DARK   = colors.HexColor('#1a0e13')
    MUTED  = colors.HexColor('#6b4a58')
    GREEN  = colors.HexColor('#16a34a')
    RED    = colors.HexColor('#dc2626')
    LIGHT  = colors.HexColor('#fce8ef')

    title_style = ParagraphStyle('Title', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=22, textColor=DARK,
        spaceAfter=4, leading=26)

    sub_style = ParagraphStyle('Sub', parent=styles['Normal'],
        fontName='Helvetica', fontSize=10, textColor=MUTED, spaceAfter=2)

    label_style = ParagraphStyle('Label', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=8, textColor=MUTED,
        spaceAfter=2, leading=10,
        textTransform='uppercase', tracking=60)

    value_style = ParagraphStyle('Value', parent=styles['Normal'],
        fontName='Helvetica', fontSize=11, textColor=DARK, spaceAfter=0)

    section_style = ParagraphStyle('Section', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=13, textColor=DARK,
        spaceBefore=14, spaceAfter=6)

    disclaimer_style = ParagraphStyle('Disclaimer', parent=styles['Normal'],
        fontName='Helvetica-Oblique', fontSize=8, textColor=MUTED,
        leading=12, spaceAfter=0)

    is_mal    = pred['prediction'] == 'Malignant'
    conf_pct  = f"{pred['confidence']*100:.1f}%"
    pred_color= RED if is_mal else GREEN

    result_style = ParagraphStyle('Result', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=28, textColor=pred_color,
        spaceAfter=4, leading=32)

    story = []

    # ── HEADER BAR (pink rule + logo)
    story.append(HRFlowable(width='100%', thickness=4, color=PINK, spaceAfter=12))

    header_data = [[
        Paragraph('<b>OncoVision AI</b>', ParagraphStyle('Logo', fontName='Helvetica-Bold',
            fontSize=18, textColor=PINK)),
        Paragraph(f'Generated: {datetime.now().strftime("%d %b %Y, %H:%M")}',
            ParagraphStyle('Date', fontName='Helvetica', fontSize=9,
                textColor=MUTED, alignment=TA_RIGHT))
    ]]
    header_table = Table(header_data, colWidths=[10*cm, 7*cm])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 4))
    story.append(Paragraph('Explainable Breast Cancer Classification — Prediction Report', sub_style))
    story.append(HRFlowable(width='100%', thickness=1, color=colors.HexColor('#ecdde4'), spaceAfter=16))

    # ── PREDICTION RESULT BOX
    result_data = [[
        Paragraph('CLASSIFICATION RESULT', label_style),
        Paragraph('CONFIDENCE', label_style),
        Paragraph('THRESHOLD', label_style),
    ],[
        Paragraph(pred['prediction'], result_style),
        Paragraph(conf_pct, ParagraphStyle('Conf', fontName='Helvetica-Bold',
            fontSize=22, textColor=pred_color, leading=28)),
        Paragraph(f"{pred['threshold']*100:.0f}%", ParagraphStyle('Thr', fontName='Helvetica',
            fontSize=22, textColor=MUTED, leading=28)),
    ]]
    result_table = Table(result_data, colWidths=[8.5*cm, 4.5*cm, 4.5*cm])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), LIGHT),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [LIGHT, LIGHT]),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#ecdde4')),
        ('ROUNDEDCORNERS', [8]),
        ('TOPPADDING',    (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING',   (0,0), (-1,-1), 14),
        ('RIGHTPADDING',  (0,0), (-1,-1), 14),
        ('LINEBELOW', (0,0), (-1,0), 0.5, colors.HexColor('#ecdde4')),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 16))

    # ── CASE DETAILS
    story.append(Paragraph('Case Details', section_style))
    story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#ecdde4'), spaceAfter=8))

    details_data = [
        [Paragraph('Patient Name', label_style),   Paragraph(pred.get('patient_name') or '—', value_style),
         Paragraph('Analysed By',  label_style),   Paragraph(pred.get('username') or '—', value_style)],
        [Paragraph('Filename',     label_style),   Paragraph(pred.get('filename') or '—', value_style),
         Paragraph('Date & Time',  label_style),   Paragraph(pred.get('created_at') or '—', value_style)],
        [Paragraph('Prediction ID',label_style),   Paragraph(prediction_id[:16]+'…', value_style),
         Paragraph('Raw Probability', label_style),Paragraph(f"{pred.get('probability', 0):.4f}", value_style)],
    ]
    details_table = Table(details_data, colWidths=[3.5*cm, 7*cm, 3.5*cm, 3.5*cm])
    details_table.setStyle(TableStyle([
        ('TOPPADDING',    (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING',   (0,0), (-1,-1), 0),
        ('RIGHTPADDING',  (0,0), (-1,-1), 8),
        ('VALIGN',        (0,0), (-1,-1), 'TOP'),
        ('LINEBELOW', (0,0), (-1,-2), 0.3, colors.HexColor('#ecdde4')),
    ]))
    story.append(details_table)
    story.append(Spacer(1, 16))

    # ── IMAGES
    def b64_to_rl_image(b64_str, w=5.5*cm, h=5.5*cm):
        """Convert base64 PNG to ReportLab Image."""
        try:
            img_bytes = base64.b64decode(b64_str)
            return RLImage(io.BytesIO(img_bytes), width=w, height=h)
        except Exception:
            return None

    def load_image_from_disk(filename, w=5.5*cm, h=5.5*cm):
        """Load saved image from history_images folder."""
        if not filename:
            return None
        path = os.path.join(HISTORY_FOLDER, filename)
        if os.path.exists(path):
            try:
                return RLImage(path, width=w, height=h)
            except Exception:
                return None
        return None

    orig_img   = load_image_from_disk(pred.get('original_image'))
    gradcam_img= load_image_from_disk(pred.get('gradcam_image'))
    shap_img   = load_image_from_disk(pred.get('shap_image'))

    placeholder_text = Paragraph('<i>(Image not available)</i>',
        ParagraphStyle('NA', fontName='Helvetica-Oblique', fontSize=9,
            textColor=MUTED, alignment=TA_CENTER))

    if orig_img or gradcam_img or shap_img:
        story.append(Paragraph('Explainability Visualisations', section_style))
        story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#ecdde4'), spaceAfter=8))

        img_row = [
            orig_img    or placeholder_text,
            gradcam_img or placeholder_text,
            shap_img    or placeholder_text,
        ]
        label_row = [
            Paragraph('Original Image',  ParagraphStyle('IL', fontName='Helvetica', fontSize=8, textColor=MUTED, alignment=TA_CENTER)),
            Paragraph('Grad-CAM Overlay',ParagraphStyle('IL', fontName='Helvetica', fontSize=8, textColor=MUTED, alignment=TA_CENTER)),
            Paragraph('SHAP Overlay',    ParagraphStyle('IL', fontName='Helvetica', fontSize=8, textColor=MUTED, alignment=TA_CENTER)),
        ]
        img_table = Table([img_row, label_row], colWidths=[5.6*cm, 5.6*cm, 5.6*cm])
        img_table.setStyle(TableStyle([
            ('ALIGN',   (0,0), (-1,-1), 'CENTER'),
            ('VALIGN',  (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING',    (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING',   (0,0), (-1,-1), 4),
            ('RIGHTPADDING',  (0,0), (-1,-1), 4),
            ('BOX',     (0,0), (-1,0), 0.5, colors.HexColor('#ecdde4')),
            ('INNERGRID',(0,0),(-1,0), 0.5, colors.HexColor('#ecdde4')),
        ]))
        story.append(img_table)
        story.append(Spacer(1, 16))

    # ── XAI DESCRIPTION
    story.append(Paragraph('Explanation Notes', section_style))
    story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#ecdde4'), spaceAfter=8))
    xai_text = (
        '<b>Grad-CAM (Gradient-weighted Class Activation Mapping):</b> Highlights tissue regions '
        'with the strongest influence on the classification decision. Warm/red areas indicate the '
        'regions the model focused on most strongly.<br/><br/>'
        '<b>SHAP (SHapley Additive exPlanations):</b> Quantifies pixel-level feature contributions '
        'using GradientExplainer. Bright regions indicate pixels with the highest absolute contribution '
        'to the prediction output.'
    )
    story.append(Paragraph(xai_text, ParagraphStyle('XAI', parent=styles['Normal'],
        fontName='Helvetica', fontSize=9, textColor=DARK, leading=14,
        backColor=LIGHT, borderPad=10, spaceAfter=0)))
    story.append(Spacer(1, 20))

    # ── FOOTER DISCLAIMER
    story.append(HRFlowable(width='100%', thickness=1, color=colors.HexColor('#ecdde4'), spaceAfter=8))
    story.append(Paragraph(
        '<b>Medical Disclaimer:</b> This report was generated by OncoVision AI, a research and educational '
        'prototype developed as a capstone project. It is NOT a validated medical diagnostic tool and '
        'must NOT replace professional pathology review. All predictions require confirmation by a '
        'qualified healthcare professional.',
        disclaimer_style
    ))

    doc.build(story)
    buf.seek(0)

    safe_name = f"OncoVision AI_Report_{(pred.get('patient_name') or 'Unknown').replace(' ','_')}_{datetime.now().strftime('%Y%m%d')}.pdf"
    return send_file(
        buf,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=safe_name
    )


@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Endpoint not found'}), 404
    return render_template_string('''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Page Not Found — OncoVision AI</title>
  <link rel="stylesheet" href="/style.css"/>
  <script src="/darkmode.js"></script>
  <style>
    .error-page { display:flex; flex-direction:column; align-items:center; justify-content:center; min-height:60vh; text-align:center; padding:3rem 1rem; }
    .error-code { font-size:6rem; font-weight:800; color:var(--pink,#c67489); line-height:1; margin-bottom:1rem; }
    .error-title { font-family:var(--font-serif,serif); font-size:2rem; margin-bottom:0.75rem; }
    .error-msg { color:var(--text-muted,#999); font-size:1rem; max-width:420px; margin-bottom:2rem; }
    .btn-home { background:var(--pink,#c67489); color:white; padding:0.85rem 2rem; border-radius:8px; text-decoration:none; font-weight:600; font-size:0.95rem; }
    .btn-home:hover { background:#b05568; }
  </style>
</head>
<body>
  <nav class="nav">
    <div class="nav__inner">
      <a href="/index.html" class="nav__logo">OncoVision AI</a>
    </div>
  </nav>
  <div class="container">
    <div class="error-page">
      <div class="error-code">404</div>
      <h1 class="error-title">Page Not Found</h1>
      <p class="error-msg">The page you're looking for doesn't exist or has been moved.</p>
      <a href="/index.html" class="btn-home">Back to Home</a>
    </div>
  </div>
</body>
</html>'''), 404

@app.errorhandler(429)
def rate_limit_exceeded(e):
    if request.path.startswith('/api/') or request.is_json:
        return jsonify({'error': 'Too many attempts. Please wait a moment before trying again.'}), 429
    return render_template_string('''<!DOCTYPE html>
<html><head><title>Too Many Requests — OncoVision AI</title>
<link rel="stylesheet" href="/style.css"/>
<style>
  .error-page { display:flex; flex-direction:column; align-items:center; justify-content:center; min-height:60vh; text-align:center; padding:3rem 1rem; }
  .error-code { font-size:6rem; font-weight:800; color:var(--pink,#c67489); line-height:1; margin-bottom:1rem; }
  .error-title { font-family:var(--font-serif,serif); font-size:2rem; margin-bottom:0.75rem; }
  .error-msg { color:var(--text-muted,#999); font-size:1rem; max-width:420px; margin-bottom:2rem; }
  .btn-home { background:var(--pink,#c67489); color:white; padding:0.85rem 2rem; border-radius:8px; text-decoration:none; font-weight:600; }
</style></head>
<body>
  <div class="error-page">
    <div class="error-code">429</div>
    <h1 class="error-title">Too Many Attempts</h1>
    <p class="error-msg">You've made too many requests. Please wait a minute and try again.</p>
    <a href="/login" class="btn-home">Back to Sign In</a>
  </div>
</body></html>'''), 429


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)