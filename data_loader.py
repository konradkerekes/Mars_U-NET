import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# -- 1. Beállítások --
# Add meg az adathalmazod elérési útját!
DATASET_PATH = "clean_dataset/"

# A képek mérete, amire átalakítjuk őket.
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256


def load_data(dataset_path):
    """
    Betölti a képeket és a maszkokat a megadott útvonalról,
    majd előkészíti őket a modell számára.
    """

    images_path = os.path.join(dataset_path, "images")
    masks_path = os.path.join(dataset_path, "masks")

    # Ellenőrzés: Létezik-e a mappa?
    if not os.path.exists(images_path) or not os.path.exists(masks_path):
        print(f"HIBA: Nem található a mappa! Ellenőrizd: {dataset_path}")
        return None, None

    image_filenames = os.listdir(images_path)

    images = []
    masks = []

    print(f"Adatok betöltése a '{dataset_path}' mappából...")

    for filename in image_filenames:
        img_full_path = os.path.join(images_path, filename)
        mask_full_path = os.path.join(masks_path, filename)

        # 1. Kép beolvasása
        image = cv2.imread(img_full_path)
        if image is None:
            continue  # Ha hibás a kép, átugorjuk

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

        # 2. Maszk beolvasása (FONTOS: Grayscale!)
        mask = cv2.imread(mask_full_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        # Átméretezés interpoláció nélkül (hogy az osztályok 0,1,2,3 maradjanak)
        mask = cv2.resize(mask, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)

        images.append(image)
        masks.append(mask)

    print(f"{len(images)} kép és maszk sikeresen betöltve.")

    if len(images) == 0:
        print("HIBA: Nem sikerült egyetlen képet sem betölteni!")
        return None, None

    # 3. NumPy tömbökké alakítás és Normalizálás
    # KÉPEK: 0-255 -> 0.0-1.0
    images_np = np.array(images, dtype=np.float32) / 255.0

    # MASZKOK: Ezek maradnak egész számok (0,1,2,3)! NEM osztunk 255-tel!
    # (N, H, W) -> (N, H, W, 1) dimenzió bővítés
    masks_np = np.expand_dims(np.array(masks, dtype=np.uint8), axis=-1)

    # 4. Adatok felosztása
    X_train, X_val, y_train, y_val = train_test_split(
        images_np, masks_np, test_size=0.2, random_state=42
    )

    return (X_train, y_train), (X_val, y_val)


def visualize_sample(images, masks, index=0):
    """
    Javított megjelenítő függvény rögzített színskálával.
    Ez segít látni a maszkot akkor is, ha az értékek kicsik (0,1,2,3).
    """

    # --- DEBUG INFO ---
    # Kiírjuk, milyen értékek vannak a maszkban.
    # Ha itt csak [0]-t látsz, akkor üres a maszk. Ha [0, 1, 2], akkor jó!
    unique_values = np.unique(masks[index])
    print(f"\n--- Minta ellenőrzése (Index: {index}) ---")
    print(f"Maszkban található osztályok: {unique_values}")

    plt.figure(figsize=(12, 6))

    # Eredeti kép
    plt.subplot(1, 2, 1)
    plt.title("Eredeti Kép")
    plt.imshow(images[index])
    plt.axis('off')

    # Maszk megjelenítése
    plt.subplot(1, 2, 2)
    plt.title("Szegmentációs Maszk (Színes)")

    # 'jet' colormap: kék(0) -> cián(1) -> sárga(2) -> piros(3)
    # vmin=0, vmax=4: Fixáljuk a skálát, hogy a 3-as érték is látszódjon
    plt.imshow(masks[index].squeeze(), cmap='jet', vmin=0, vmax=4, interpolation='nearest')
    plt.axis('off')

    plt.show()


# -- Fő programrész (Teszteléshez) --
if __name__ == '__main__':
    # Adatok betöltése
    (train_images, train_masks), (val_images, val_masks) = load_data(DATASET_PATH)

    if train_images is not None:
        # Kiírjuk az adatok formáját ellenőrzésképpen
        print("\nAdatok formátuma:")
        print(f"Tanító képek: {train_images.shape}")
        print(f"Tanító maszkok: {train_masks.shape}")

        # Megjelenítünk egy mintát (próbálj ki többet is, pl. index=10, 25)
        print("\nEgy minta megjelenítése a tanító adathalmazból...")
        visualize_sample(train_images, train_masks, index=5)