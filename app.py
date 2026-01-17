import tensorflow as tf
import numpy as np
import cv2
import base64
import os
import glob
from flask import Flask, request, jsonify
from flask_cors import CORS

# -- Globális Beállítások --
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
NUM_CLASSES = 4

# Osztálynevek a JSON válaszhoz (Angol)
CLASS_NAMES = {
    0: "Soil / Background",
    1: "Rock",
    2: "Sand",
    3: "Other"
}


# -- OKOS MODELL BETÖLTÉS --
# Ez a rész automatikusan megkeresi a legfrissebb .h5 fájlt a mappában!
def load_latest_model():
    # Megkeressük az összes .h5 fájlt
    list_of_files = glob.glob(os.path.join('models', '*.h5'))

    if not list_of_files:
        print("HIBA: Nem található .h5 modellfájl a mappában!")
        return None, None

    # Kiválasztjuk a legutóbb módosítottat (vagyis a legfrissebb tanítást)
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Legfrissebb modell betöltése: {latest_file}...")

    try:
        model = tf.keras.models.load_model(latest_file)
        print("Modell sikeresen betöltve!")
        model_name_only = os.path.basename(latest_file)
        return model, model_name_only
    except Exception as e:
        print(f"Hiba a modell betöltésekor: {e}")
        return None, None


# Itt hívjuk meg a betöltést
model, model_name = load_latest_model()


# -- Képfeldolgozó Függvények --
def preprocess_image(image_bytes):
    """
    Kép előkészítése: Átméretezés és Normalizálás.
    FONTOS: Itt is osztunk 255-tel, ahogy a tanításnál!
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        return None, None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (IMAGE_WIDTH, IMAGE_HEIGHT))

    # Normalizálás (0-1 közé)
    image_normalized = image_resized / 255.0

    # Batch dimenzió hozzáadása (1, 256, 256, 3)
    input_image = np.expand_dims(image_normalized, axis=0)

    return image_resized, input_image


def create_visual_mask(mask):
    """
    A 0,1,2,3 számokból színes képet csinál.
    Színek (RGB formátumban):
    """
    visual_mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)

    COLORS = {
        0: [0, 0, 255],  # Class 0 (Soil)  = Kék
        1: [255, 0, 0],  # Class 1 (Rock)  = Piros
        2: [0, 255, 0],  # Class 2 (Sand)  = Zöld
        3: [255, 255, 0]  # Class 3 (Other) = Sárga
    }

    for class_id, color in COLORS.items():
        # Ahol a maszk értéke megegyezik az osztály ID-jával, oda színezzük
        visual_mask[mask == class_id] = color

    return visual_mask


# -- Flask App Inicializálása --
app = Flask(__name__)
# Engedélyezzük a CORS-t, hogy a frontend (HTML) elérje a backendet
CORS(app)


# -- API Végpont --
@app.route('/predict', methods=['POST'])
def predict():
    # Biztonsági ellenőrzés
    if model is None:
        return jsonify({"error": "A modell nincs betöltve a szerveren!"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "Nem küldtél fájlt!"}), 400

    file = request.files['file']
    image_bytes = file.read()

    # 1. Előkészítés
    original_resized, processed_image = preprocess_image(image_bytes)
    if processed_image is None:
        return jsonify({"error": "Érvénytelen képfájl"}), 400

    # 2. Predikció (A modell valószínűségeket ad)
    predicted_probs = model.predict(processed_image)
    # Visszaalakítjuk osztályokká (0, 1, 2, 3)
    predicted_mask = np.argmax(predicted_probs, axis=-1)[0]

    # 3. Statisztika számítása
    stats = {}
    total_pixels = predicted_mask.size
    for class_id, name in CLASS_NAMES.items():
        pixel_count = np.sum(predicted_mask == class_id)
        percent = (pixel_count / total_pixels) * 100
        stats[name] = round(percent, 2)  # Kerekítés 2 tizedesre

    # 4. Válaszkép (Maszk) generálása
    visual_mask_img = create_visual_mask(predicted_mask)

    # Átalakítás Base64 formátummá, hogy visszaküldhessük a böngészőnek
    # Fontos: Az OpenCV BGR-t használ mentésnél, a visual_mask viszont RGB.
    # Ezért konvertálunk RGB->BGR-be kódolás előtt.
    _, buffer = cv2.imencode('.png', cv2.cvtColor(visual_mask_img, cv2.COLOR_RGB2BGR))
    mask_base64 = base64.b64encode(buffer).decode('utf-8')

    # JSON válasz küldése
    return jsonify({
        "model_name": model_name,  # Visszaküldjük, melyik modellt használta
        "statistics": stats,
        "mask_image": f"data:image/png;base64,{mask_base64}"
    })


# Szerver indítása
if __name__ == '__main__':
    print("Flask szerver indítása...")
    # debug=True segít a hibák megtalálásában fejlesztés közben
    app.run(debug=True, port=5000)