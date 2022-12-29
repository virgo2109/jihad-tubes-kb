# -- coding: utf-8 --

import os

## Data Understanding

# Untuk memahami dataset berupa gambar, saya mendefinisikan terlebih dahulu direktori datasetnya ke dalam sebuah variabel. Selanjutnya tiap kelas atau label yang tersedia dapat diketahui dengan memanggil `os.listdir`

# Mendefinisikan direktori utama dataset
base_dir = 'D:\SEMESTER 3\Kecerdasan Buatan\KB yun\KB\Dataset Tubes Mobil Mazda'

print(os.listdir(base_dir))

"""Untuk mengetahui jumlah file yang ada, dapat menggunakan fungsi `len()` pada tiap direktori label dan menjumlahkannya. Untuk memudahkan pembacaan data dapat juga divisualisasikan menggunakan bantuan library `matplotlib`."""

# Menghitung jumlah gambar pada dataset
number_label = {}
total_files = 0
for i in os.listdir(base_dir):
    counting = len(os.listdir(os.path.join(base_dir, i)))
    number_label[i] = counting
    total_files += counting

print("Total Files : " + str(total_files))

# Visualisasi jumlah gambar tiap kelas
import matplotlib.pyplot as plt

plt.bar(number_label.keys(), number_label.values());
plt.title("Jumlah Gambar Tiap Label");
plt.xlabel('Label');
plt.ylabel('Jumlah Gambar');

"""Dari diagram batang di atas dapat diketahui bahwa gambar yang ada pada setiap label memiliki jumlah yang berbeda-beda. Semua label memiliki jumlah lebih dari 700 gambar. Jumlah gambar paling banyak ada pada bunga Dandelion, sedangkan yang paling sedikit adalah bunga matahari (Sunflower). Namun perbedaan jumlah gambar tidak terlalu jauh dan tidak perlu diseimbangkan."""

# Menampilkan sampel gambar tiap kelas
import matplotlib.image as mpimg

img_each_class = 1
img_samples = {}
classes = list(number_label.keys())

for c in classes:
    temp = os.listdir(os.path.join(base_dir, c))[:img_each_class]
    for item in temp:
        img_path = os.path.join(base_dir, c, item)
        img_samples[c] = img_path

for i in img_samples:
    fig = plt.gcf()
    img = mpimg.imread(img_samples[i])
    plt.title(i)
    plt.imshow(img)
    plt.show()

"""Dari beberapa gambar di atas dapat diketahui bahwa gambar yang tersedia pada dataset ini memiliki ukuran yang berbeda-beda.

## Data Preparation

Setelah memahami data, selanjutnya adalah mempersiapkan data sebelum nantinya masuk ke modelling. Penyiapan ini termasuk didalamnya adalah pembagian data (split) menjadi data latih dan validasi. Pembagian data ini diperlukan sebelum nantinya digunakan untuk melatih model yang dibuat serta menghitung akurasi modelnya.
"""

IMAGE_SIZE = (200, 200)
BATCH_SIZE = 32
SEED = 999

# Menggunakan ImageDataGenerator untuk preprocessing
import tensorflow as tf

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    validation_split=0.2
)

# Menyiapkan data train dan data validation
train_data = datagen.flow_from_directory(
    base_dir,
    class_mode='categorical',
    subset='training',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)

valid_data = datagen.flow_from_directory(
    base_dir,
    class_mode='categorical',
    subset='validation',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)

"""Selain membagi data, akan diterapkan juga image augmentation. Hal ini diterapkan berdasarkan data gambar yang telah ditampilkan sebelumnya. Image augmentation yang dilakukan
 di sini menggunakan layer RandomFlip, RandomRotation, RandomZoom serta Rescaling pada gambar.
"""

# Image Augmentation
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal",
                                   input_shape=(IMAGE_SIZE[0],
                                                IMAGE_SIZE[1],
                                                3)),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.Rescaling(1. / 255)
    ]
)

"""## Modelling

### Membuat Arsitektur CNN

#### Penyusunan Layer
"""

# Membuat arsitektur model CNN
cnn_model = tf.keras.models.Sequential([
    data_augmentation,
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

# Compiling model
cnn_model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

"""#### Melatih Model CNN"""

# Training model CNN
cnn_hist = cnn_model.fit(
    train_data,
    epochs=20,
    validation_data=valid_data
)

"""#### Evaluasi Model CNN"""

# Membuat plot akurasi model CNN
plt.figure(figsize=(10, 4))
plt.plot(cnn_hist.history['accuracy'])
plt.plot(cnn_hist.history['val_accuracy'])
plt.title('CNN Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True)
plt.show()

print()

# Membuat plot loss model CNN
plt.figure(figsize=(10, 4))
plt.plot(cnn_hist.history['loss'])
plt.plot(cnn_hist.history['val_loss'])
plt.title('CNN Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True)
plt.show()

"""### Transfer Learning Menggunakan VGG16

#### Memuat Model VGG16
"""
import keras
import keras_applications
import tensorflow as tf
from keras.applications.vgg16 import VGG16

## Loading VGG16 model
base_vgg_model = VGG16(weights="imagenet", include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
base_vgg_model.trainable = False

# Preprocessing Input
vgg_preprocess = tf.keras.applications.vgg16.preprocess_input
train_data.preprocessing_function = vgg_preprocess

# Transfer learning dengan VGG16
vgg_model = tf.keras.models.Sequential([
    data_augmentation,
    base_vgg_model,
    tf.keras.layers.Dropout(0.7),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

# Compiling model
vgg_model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

"""#### Melatih Model"""

# Melatih model VGG16
vgg_hist = vgg_model.fit(
    train_data,
    epochs=20,
    validation_data=valid_data
)

"""#### Evaluasi Model"""

# Membuat plot akurasi model VGG16
plt.figure(figsize=(10, 4))
plt.plot(vgg_hist.history['accuracy'])
plt.plot(vgg_hist.history['val_accuracy'])
plt.title('VGG16 Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True)
plt.show()

print()

# Membuat plot loss model VGG16
plt.figure(figsize=(10, 4))
plt.plot(vgg_hist.history['loss'])
plt.plot(vgg_hist.history['val_loss'])
plt.title('VGG16 Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True)
plt.show()