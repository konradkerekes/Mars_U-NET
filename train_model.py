import os
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# -- 1. Globális Beállítások --
DATASET_PATH = "clean_dataset/"
CHECKPOINT_DIR = 'checkpoints/'
MODELS_DIR = 'models/'
LOG_FILE = "training_log.csv"

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_CHANNELS = 3
NUM_CLASSES = 4
BATCH_SIZE = 16
TOTAL_EPOCHS = 25


# -- 2. A "MarsDataGenerator" Osztály --
class MarsDataGenerator(Sequence):
    """
    Okos adagoló, ami helyesen tölti be a képeket és maszkokat.
    """

    def __init__(self, image_filenames, mask_filenames, batch_size, image_dir, mask_dir):
        self.image_filenames = image_filenames
        self.mask_filenames = mask_filenames
        self.batch_size = batch_size
        self.image_dir = image_dir
        self.mask_dir = mask_dir

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / self.batch_size))

    def __getitem__(self, index):
        batch_img_files = self.image_filenames[index * self.batch_size:(index + 1) * self.batch_size]
        batch_mask_files = self.mask_filenames[index * self.batch_size:(index + 1) * self.batch_size]

        images = np.zeros((len(batch_img_files), IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=np.float32)
        masks = np.zeros((len(batch_mask_files), IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.uint8)

        for i, filename in enumerate(batch_img_files):
            # Kép betöltése és normalizálása
            img = cv2.imread(os.path.join(self.image_dir, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
            images[i] = img / 255.0  # Képet osztjuk 255-tel

            # Maszk betöltése (Grayscale!) és NEM normalizálása
            mask = cv2.imread(os.path.join(self.mask_dir, filename), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)

            # Biztonsági tisztítás (hogy ne legyenek 4,5,stb. osztályok véletlenül se)
            mask[mask >= NUM_CLASSES] = 0

            masks[i] = np.expand_dims(mask, axis=-1)

        return images, tf.keras.utils.to_categorical(masks, num_classes=NUM_CLASSES)


# -- 3. U-Net Modell Építő Függvény --
def build_unet_model(input_shape, num_classes):
    inputs = Input(input_shape)

    # Encoder
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    # Bottom
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)

    # Decoder
    u4 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c3)
    u4 = concatenate([u4, c2])
    c4 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u4)
    c4 = Dropout(0.1)(c4)
    c4 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)

    u5 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = concatenate([u5, c1])
    c5 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u5)
    c5 = Dropout(0.1)(c5)
    c5 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c5)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


# -- 4. Fő Programrész --
if __name__ == '__main__':
    print("\n--- MARS MODELL TANÍTÁS INDÍTÁSA ---")
    print(f"Cél: A modell eljuttatása a(z) {TOTAL_EPOCHS}. epochig.")

    images_dir = os.path.join(DATASET_PATH, "images")
    masks_dir = os.path.join(DATASET_PATH, "masks")

    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        print(f"HIBA: Nem találom a mappákat itt: {DATASET_PATH}")
        print("Tipp: Futtasd le először a preprocess_data.py-t!")
        exit()

    all_filenames = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    if len(all_filenames) == 0:
        print("HIBA: A 'clean_dataset' mappa üres!")
        exit()

    # -- C. Adatok szétválasztása és Generátorok --
    train_filenames, val_filenames = train_test_split(all_filenames, test_size=0.2, random_state=42)
    print(f"Adatok: {len(train_filenames)} tanító kép | {len(val_filenames)} validációs kép")

    training_generator = MarsDataGenerator(train_filenames, train_filenames, BATCH_SIZE, images_dir, masks_dir)
    validation_generator = MarsDataGenerator(val_filenames, val_filenames, BATCH_SIZE, images_dir, masks_dir)

    # -- D. Modell létrehozása --
    model = build_unet_model((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), NUM_CLASSES)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # -- E. Checkpoint & Előzmények Kezelése --
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    latest_checkpoint_path = None
    latest_epoch = 0

    # Megkeressük a legfrissebb mentett súlyt a checkpoints mappában
    if os.path.exists(CHECKPOINT_DIR):
        checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.weights.h5')]
        files_with_epochs = []
        for f in checkpoint_files:
            try:
                # Pl. checkpoint-05.weights.h5 -> 5
                ep = int(f.split('-')[1].split('.')[0])
                files_with_epochs.append((ep, f))
            except:
                pass

        if files_with_epochs:
            latest_epoch, latest_file = max(files_with_epochs, key=lambda x: x[0])
            latest_checkpoint_path = os.path.join(CHECKPOINT_DIR, latest_file)

    # Ha teljesen elölről kezdjük (nincs checkpoint), töröljük a régi log fájlt
    if latest_checkpoint_path is None and os.path.exists(LOG_FILE):
        print("Új tanítás kezdődik, a régi statisztikai log fájlt töröljük.")
        os.remove(LOG_FILE)

    if latest_checkpoint_path:
        print(f"Súlyok betöltése innen: {latest_checkpoint_path}")
        print(f"Tanítás FOLYTATÁSA a(z) {latest_epoch}. epochtól.")
        model.load_weights(latest_checkpoint_path)
    else:
        print("Nincs mentés. Tanítás a nulláról.")

    # -- F. Callbackek beállítása --
    checkpoint_filepath = os.path.join(CHECKPOINT_DIR, 'checkpoint-{epoch:02d}.weights.h5')

    csv_logger = CSVLogger(LOG_FILE, append=True)  # append=True a folytatáshoz

    callbacks = [
        ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, monitor='val_loss', mode='min',
                        save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1),
        csv_logger
    ]

    # -- G. Tanítás Indítása (Csak ha még nem értük el a célt) --
    if latest_epoch < TOTAL_EPOCHS:
        print(f"\nTanítás indítása...")
        history = model.fit(
            training_generator,
            validation_data=validation_generator,
            epochs=TOTAL_EPOCHS,
            initial_epoch=latest_epoch,  # Ez mondja meg, honnan folytassa a számlálást
            verbose=1,
            callbacks=callbacks
        )
        print("Tanítás kész.")
    else:
        print(
            f"\nA modell már elérte a {latest_epoch} epochot (Cél: {TOTAL_EPOCHS}). Nincs szükség további tanításra.")

    # -- H. Mentés a 'models' mappába --
    os.makedirs(MODELS_DIR, exist_ok=True)  # Mappa létrehozása, ha nincs

    model_filename = f"mars_model_{TOTAL_EPOCHS}_epochs.h5"
    model_save_path = os.path.join(MODELS_DIR, model_filename)

    print(f"Modell mentése ide: {model_save_path} ...")
    model.save(model_save_path)
    print(f"SIKERES MENTÉS! A fájl készen áll a használatra.")

    # -- I. VIZUALIZÁCIÓ 1: Grafikon --
    print("\nTeljes tanítási történelem betöltése...")
    if os.path.exists(LOG_FILE):
        try:
            log_data = pd.read_csv(LOG_FILE)
            plt.figure(figsize=(12, 5))

            # Pontosság
            plt.subplot(1, 2, 1)
            plt.plot(log_data['accuracy'], label='Tanítás')
            plt.plot(log_data['val_accuracy'], label='Validáció')
            plt.title(f'Modell Pontossága ({len(log_data)} epoch)')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)

            # Veszteség
            plt.subplot(1, 2, 2)
            plt.plot(log_data['loss'], label='Tanítás')
            plt.plot(log_data['val_loss'], label='Validáció')
            plt.title('Modell Vesztesége')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Nem sikerült a grafikont kirajzolni: {e}")

    # -- J. VIZUALIZÁCIÓ 2: Kép és Maszk ellenőrzés --
    print("\nEgy tesztkép ellenőrzése...")
    test_images_batch, original_masks_batch_one_hot = validation_generator[0]

    test_image_to_predict = test_images_batch[0]
    original_mask = np.argmax(original_masks_batch_one_hot[0], axis=-1)

    input_for_model = np.expand_dims(test_image_to_predict, axis=0)
    predicted_mask_probs = model.predict(input_for_model)
    predicted_mask = np.argmax(predicted_mask_probs, axis=-1)[0]

    plt.figure(figsize=(15, 6))

    plt.subplot(1, 3, 1)
    plt.title("Eredeti Kép")
    plt.imshow(test_image_to_predict)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Eredeti Maszk")
    plt.imshow(original_mask, cmap='jet', vmin=0, vmax=NUM_CLASSES)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Modell Predikció")
    plt.imshow(predicted_mask, cmap='jet', vmin=0, vmax=NUM_CLASSES)
    plt.axis('off')

    plt.tight_layout()
    plt.show()