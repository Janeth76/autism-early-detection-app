# Importar librerías
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Cargar dataset
data = pd.read_csv("autism_dataset_simulated.csv")

# Separar variables
X = data.drop("riesgo_autismo", axis=1)
y = data["riesgo_autismo"]

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Escalar datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear modelo de red neuronal
model = Sequential()

model.add(Dense(32, activation="relu", input_shape=(X_train.shape[1],)))
model.add(Dropout(0.3))

model.add(Dense(16, activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(8, activation="relu"))

model.add(Dense(1, activation="sigmoid"))

# Compilar modelo
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Entrenar modelo
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2
)

# Evaluar modelo
loss, accuracy = model.evaluate(X_test, y_test)

print("Accuracy:", accuracy)

# Predicción de ejemplo
nuevo_nino = np.array([[0,0,0,0,0,1,0,1]])
nuevo_nino = scaler.transform(nuevo_nino)

prediction = model.predict(nuevo_nino)

if prediction > 0.5:
    print("Alto riesgo de autismo")
else:
    print("Bajo riesgo de autismo")
