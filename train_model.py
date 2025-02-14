import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split
import pickle

dataset_dir = r"C:\Users\omarm\Documents\mypathon"
cats_dir = os.path.join(dataset_dir, "cats")
dogs_dir = os.path.join(dataset_dir, "dogs")

def load_images(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = tf.keras.preprocessing.image.load_img(os.path.join(folder, filename), target_size=(150, 150))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = img / 255.0
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)

cat_images, cat_labels = load_images(cats_dir, 0)
dog_images, dog_labels = load_images(dogs_dir, 1)

X = np.concatenate((cat_images, dog_images))
y = np.concatenate((cat_labels, dog_labels))

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),  # Reduced number of units
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

batch_size = 32
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(X_train) // batch_size,
    epochs=20,
    validation_data=(X_val, y_val)
)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Save the full model using pickle
with open("cat_dog_classifier.pkl", "wb") as f:
    pickle.dump(model, f)