import os
import cv2

# -- 1. Beállítások --
# Add meg a letöltött és kicsomagolt adathalmaz mappájának útvonalát!
# Ez a mappa tartalmazza a különböző (pl. msl, m2020) almappákat.
RAW_DATA_PATH = "raw_dataset/"

# Ide fogjuk menteni a tiszta, használatra kész adatokat.
CLEAN_DATA_PATH = "clean_dataset"


def clean_and_organize_data(raw_path, clean_path):
    """
    Végigmegy a nyers adatokon, megkeresi az érvényes kép-maszk párokat,
    és átmásolja őket egy új, tiszta mappastruktúrába.
    """

    # Létrehozzuk a célmappákat, ha még nem léteznek
    clean_images_path = os.path.join(clean_path, "images")
    clean_masks_path = os.path.join(clean_path, "masks")
    os.makedirs(clean_images_path, exist_ok=True)
    os.makedirs(clean_masks_path, exist_ok=True)

    print(f"Adatok tisztítása a '{raw_path}' mappából...")
    print(f"A tiszta adatok ide kerülnek: '{clean_path}'")

    # Szótárak a fájlok tárolására (név -> teljes útvonal)
    images_map = {}
    labels_map = {}

    # 1. Fájlok összegyűjtése
    for root, dirs, files in os.walk(raw_path):
        for file in files:
            # A fájlnév és a kiterjesztés szétválasztása
            filename_no_ext, ext = os.path.splitext(file)
            ext = ext.lower()

            # Csak a releváns képfájlokat vesszük figyelembe
            if ext in ['.jpg', '.jpeg', '.png']:
                full_path = os.path.join(root, file)
                # Megnézzük, hogy a fájl egy kép vagy egy maszk-e az útvonala alapján
                if 'label' in root or 'mask' in root:
                    labels_map[filename_no_ext] = full_path
                else:
                    images_map[filename_no_ext] = full_path

    print(f"Összesen {len(images_map)} kép és {len(labels_map)} címke található.")

    # 2. Párosítás, ellenőrzés és másolás
    processed_count = 0
    skipped_count = 0

    for img_name, img_path in images_map.items():
        # Van-e ehhez a képhez tartozó maszk?
        if img_name in labels_map:
            mask_path = labels_map[img_name]

            # Ellenőrizzük, hogy a maszk fájl nem üres-e (legalább 1KB)
            if os.path.getsize(mask_path) > 1024:
                # Új, egységes nevet adunk a fájloknak
                new_filename = f"mars_image_{processed_count}.png"

                # Cél útvonalak
                dest_img_path = os.path.join(clean_images_path, new_filename)
                dest_mask_path = os.path.join(clean_masks_path, new_filename)

                # --- ITT A JAVÍTÁS ---
                try:
                    # KÉP: Hagyományos beolvasás (színes)
                    img = cv2.imread(img_path)

                    # MASZK: FONTOS! IMREAD_GRAYSCALE használata
                    # Ez biztosítja, hogy a 0,1,2,3 értékek nyers formában maradjanak
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                    if img is not None and mask is not None:
                        # Mentés veszteségmentes PNG formátumban
                        cv2.imwrite(dest_img_path, img)
                        cv2.imwrite(dest_mask_path, mask)
                        processed_count += 1
                    else:
                        skipped_count += 1
                except Exception as e:
                    print(f"Hiba a '{img_path}' fájl feldolgozásakor: {e}")
                    skipped_count += 1
            else:
                skipped_count += 1
        else:
            skipped_count += 1

    print("\nElőfeldolgozás befejezve!")
    print(f"Sikeresen feldolgozott és átmásolt párok: {processed_count}")
    print(f"Kihagyott vagy hibás fájlok: {skipped_count}")


# -- Fő programrész --
if __name__ == '__main__':
    clean_and_organize_data(RAW_DATA_PATH, CLEAN_DATA_PATH)