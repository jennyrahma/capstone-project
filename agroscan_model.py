import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Input, Concatenate
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# =========================
# 1. Dataset Lingkungan
# =========================

# Load dataset lingkungan
df_env = pd.read_csv('indonesia_environment_data.csv')

# Preprocessing dataset lingkungan
X_env = df_env[['Temperature', 'Humidity', 'Soil pH', 'Rainfall', 'Light Intensity']].values
y_env = df_env['Health Score'].values

# Normalisasi data
scaler = MinMaxScaler()
X_env = scaler.fit_transform(X_env)

# Split data
X_env_train, X_env_test, y_env_train, y_env_test = train_test_split(X_env, y_env, test_size=0.2, random_state=42)

# =========================
# 2. Dataset Citra Daun
# =========================

# Augmentasi data citra daun
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Load data citra
train_generator = datagen.flow_from_directory(
    'leaf_images/',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',  # Binary classification: Healthy vs Diseased
    subset='training'
)

val_generator = datagen.flow_from_directory(
    'leaf_images/',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# =========================
# 3. Model Lingkungan
# =========================

# Model Dense untuk data lingkungan
env_input = Input(shape=(X_env.shape[1],))
x = Dense(64, activation='relu')(env_input)
x = Dense(32, activation='relu')(x)
env_output = Dense(1, activation='linear', name='env_output')(x)

# =========================
# 4. Model Citra Daun
# =========================

# Model CNN untuk citra daun
image_input = Input(shape=(150, 150, 3))
y = Conv2D(32, (3, 3), activation='relu')(image_input)
y = MaxPooling2D((2, 2))(y)
y = Conv2D(64, (3, 3), activation='relu')(y)
y = MaxPooling2D((2, 2))(y)
y = Flatten()(y)
y = Dense(128, activation='relu')(y)
y = Dropout(0.5)(y)
image_output = Dense(1, activation='sigmoid', name='image_output')(y)

# =========================
# 5. Gabungkan Model
# =========================

# Gabungkan kedua model
combined = Concatenate()([env_output, image_output])
z = Dense(64, activation='relu')(combined)
final_output = Dense(1, activation='linear', name='final_output')(z)

# Model akhir
model = Model(inputs=[env_input, image_input], outputs=final_output)

# Compile model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# =========================
# 6. Latih Model
# =========================

# Buat generator untuk data lingkungan
def environment_generator(X, y, batch_size=32):
    while True:
        idx = np.random.randint(0, X.shape[0], batch_size)
        yield X[idx], y[idx]

env_train_gen = environment_generator(X_env_train, y_env_train)
env_val_gen = environment_generator(X_env_test, y_env_test)

# Latih model
model.fit(
    x=[env_train_gen, train_generator],
    validation_data=([env_val_gen, val_generator]),
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    epochs=10
)

# =========================
# 7. Evaluasi Model
# =========================

# Evaluasi model
loss, mae = model.evaluate([X_env_test, val_generator])
print("Mean Absolute Error:", mae)

import matplotlib.pyplot as plt

# Prediksi pada data test
y_pred_test = model.predict([X_env_test, val_generator])

# Plot prediksi vs nilai aktual
plt.figure(figsize=(8, 8))
plt.scatter(y_env_test, y_pred_test, color='blue', label='Predicted vs Actual')
plt.plot([0, 1], [0, 1], '--', color='red', label='Ideal Prediction')  # Garis referensi
plt.xlabel('Actual Health Score')
plt.ylabel('Predicted Health Score')
plt.title('Actual vs Predicted Health Score')
plt.legend()
plt.grid()
plt.show()

import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Contoh data lingkungan baru
new_env_data = np.array([[30, 80, 6.5, 100, 3000]])  # Temperature, Humidity, Soil pH, Rainfall, Light Intensity
new_env_data = scaler.transform(new_env_data)  # Normalisasi data lingkungan

# Contoh citra daun baru
image_path = 'leaf_images/Healthy/healthy_leaf1.jpg'  # Ganti dengan path gambar daun Anda
img = load_img(image_path, target_size=(150, 150))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Prediksi skor kesehatan
predicted_health = model.predict([new_env_data, img_array])
print(f"Predicted Health Score for new data: {predicted_health[0][0]}")
