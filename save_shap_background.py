"""
save_shap_background.py
=======================
Run this ONCE in Google Colab to save 50 background training images
that the Flask backend needs for SHAP explanations.

Copy the output file (shap_background.npy) into the same folder as app.py.

Usage (in Colab, after your notebook sections have run):
    exec(open('save_shap_background.py').read())
"""

import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input

# ── Uses variables already defined in your notebook ──────────────────
# train_df, IMG_SIZE, load_img_array (or equivalent)

N = 50
bg_sample = train_df.sample(N, random_state=42)
bg_images = []

for path in bg_sample['filepath'].values:
    img = tf.keras.preprocessing.image.load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = tf.keras.preprocessing.image.img_to_array(img)
    bg_images.append(arr)

bg_images = np.array(bg_images, dtype=np.float32)
bg_images = preprocess_input(bg_images)

save_path = '/content/drive/MyDrive/EfficientNet/outputs_efficientnet/shap_background.npy'
np.save(save_path, bg_images)
print(f'Saved {N} background images to: {save_path}')
print(f'Shape: {bg_images.shape}')
print('Download this file and place it next to app.py')
