import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tkinter as tk
from tkinter import filedialog
import os

# -- 1. Globális Beállítások --
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
NUM_CLASSES = 4

# Osztályok nevei és színei (BGR a rajzoláshoz, RGB a legendhez majd konvertálva lesz)
CLASS_INFO = {
    0: {"name": "Talaj / Háttér", "color": (255, 0, 0)},  # Kék (BGR)
    1: {"name": "Kőzet", "color": (0, 0, 255)},  # Piros (BGR)
    2: {"name": "Homok", "color": (0, 255, 0)},  # Zöld (BGR)
    3: {"name": "Egyéb / Rover", "color": (0, 255, 255)}  # Sárga (BGR)
}


# -- 2. Segédfüggvények --

def select_file(title, filetypes):
    """Fájlválasztó ablak megnyitása."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return file_path


def preprocess_image(image_path):
    """
    Beolvassa és előkészíti a képet.
    FONTOS: Itt ugyanazt a normalizálást használjuk (osztás 255-tel), mint a train_model.py-ban!
    """
    image = cv2.imread(image_path)
    if image is None:
        return None, None, None

    # Eredeti kép mentése (RGB-ben a megjelenítéshez)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (IMAGE_WIDTH, IMAGE_HEIGHT))

    # Normalizálás: 0-1 közé (ez kell a modellnek)
    image_normalized = image_resized / 255.0

    # Batch dimenzió hozzáadása: (1, 256, 256, 3)
    input_image = np.expand_dims(image_normalized, axis=0)

    return image, image_resized, input_image


def calculate_statistics(mask):
    """Statisztika számítása százalékban."""
    stats = {}
    total_pixels = mask.size
    for class_id, info in CLASS_INFO.items():
        pixel_count = np.sum(mask == class_id)
        percentage = (pixel_count / total_pixels) * 100
        stats[info['name']] = percentage
    return stats


# -- 3. Fő Programrész --
if __name__ == '__main__':
    # --- 1. Modell Betöltése ---
    print("Kérlek, válaszd ki a .h5 modellfájlt...")
    model_path = select_file("Modell választása", [("Keras modellek", "*.h5")])

    if not model_path:
        print("Nem választottál modellt. Kilépés.")
        exit()

    try:
        print(f"Modell betöltése: {os.path.basename(model_path)}...")
        model = tf.keras.models.load_model(model_path)
        print("Modell sikeresen betöltve.")
    except Exception as e:
        print(f"Hiba: {e}")
        exit()

    # --- 2. Kép Betöltése ---
    print("\nKérlek, válaszd ki a képet...")
    image_path = select_file("Kép választása", [("Képek", "*.png *.jpg *.jpeg")])

    if not image_path:
        print("Nem választottál képet. Kilépés.")
        exit()

    # --- 3. Előkészítés és Predikció ---
    original_bgr, display_image, input_tensor = preprocess_image(image_path)

    if input_tensor is None:
        print("Hiba a kép beolvasásakor.")
        exit()

    print("Predikció folyamatban...")
    # A modell valószínűségeket ad vissza
    predicted_probs = model.predict(input_tensor)
    # Visszaalakítjuk osztályokká (0, 1, 2, 3)
    predicted_mask = np.argmax(predicted_probs, axis=-1)[0]

    # --- 4. Vizualizáció (Kombinált nézet) ---

    # A) Kontúros kép készítése (Overlay)
    overlay_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)  # Vissza BGR-be a rajzoláshoz
    stats = calculate_statistics(predicted_mask)

    for class_id, info in CLASS_INFO.items():
        # Csak akkor rajzolunk, ha van ilyen osztály a képen
        if stats[info['name']] > 0.0:
            class_mask = np.zeros(predicted_mask.shape, dtype=np.uint8)
            class_mask[predicted_mask == class_id] = 255

            # Kontúrok keresése és rajzolása
            contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_image, contours, -1, info['color'], 2)

    overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)  # Vissza RGB-be a megjelenítéshez

    # B) Plot elkészítése
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    # Bal oldal: Szép kontúros kép
    ax[0].imshow(overlay_image)
    ax[0].set_title("Eredmény Kontúrokkal")
    ax[0].axis('off')

    # Jobb oldal: Nyers Maszk (mint a tanításnál!)
    # Itt használjuk a 'jet' colormap-et és a fix skálát (vmin=0, vmax=4)
    im = ax[1].imshow(predicted_mask, cmap='jet', vmin=0, vmax=NUM_CLASSES, interpolation='nearest')
    ax[1].set_title("Nyers Maszk (Osztályok)")
    ax[1].axis('off')

    # Legend (Jelmagyarázat) készítése
    legend_patches = []
    stats_text = "Statisztika:\n"

    for class_id, info in CLASS_INFO.items():
        # Matplotlib RGB-t vár, de mi BGR-t adtunk meg fent, ezért fordítjuk: [::-1]
        color_rgb = np.array(info['color'])[::-1] / 255.0
        legend_patches.append(mpatches.Patch(color=color_rgb, label=info['name']))
        stats_text += f"{info['name']}: {stats[info['name']]:.2f}%\n"

    # Szövegdoboz a statisztikával
    fig.text(0.02, 0.02, stats_text, fontsize=11,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Jelmagyarázat a kép szélére
    fig.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(0.98, 0.95))

    plt.suptitle(f"Modell: {os.path.basename(model_path)}", fontsize=14)
    plt.tight_layout()
    plt.show()