import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical

# --- Charger les données ---
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

X = train.drop("label", axis=1).values.reshape(-1, 28, 28, 1) / 255.0
y = to_categorical(train["label"].values, num_classes=10)
X_test = test.values.reshape(-1, 28, 28, 1) / 255.0

# --- Modèle CNN ---
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# --- Entraînement ---
model.fit(X, y, epochs=10, batch_size=128, validation_split=0.1)

# --- Prédictions ---
pred = np.argmax(model.predict(X_test), axis=1)

# --- Submission ---
submission = pd.DataFrame({"ImageId": np.arange(1, len(pred)+1), "Label": pred})
submission.to_csv("submission.csv", index=False)
print("✅ Fichier submission.csv généré !")
