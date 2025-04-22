import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from keras.models import Sequential
from keras.layers import GRU, Dense, Input, Dropout


class Model():
    
    def __init__(self, X, y):
        """
        Constructor de la clase Model.
        
        - X: Datos de entrada.
        - y: Datos de salida.
        """
        
        self.X = X.T.shape
        self.y = y.T.shape
        
        self.model = self.build_model()

    def build_model(self):
        """
        Método para construir y compilar el modelo.
        
        - X: Datos de entrada.
        - y: Datos de salida.
        """
        # Definición del modelo
        input_shape = self.X
        output_shape = self.y[0]

        model = Sequential([
            Input(shape=input_shape),  # Capa de entrada
            GRU(64, return_sequences=True, activation='relu'),  # GRU con 64 unidades
            Dense(32, activation='relu'),  # Capa densa con 32 unidades
            Dense(16, activation='relu'),  # Capa densa con 16 unidades
            Dense(output_shape, activation='linear')  # Capa de salida con activación lineal
        ])

        model.compile(
            optimizer='adam',
            loss='mse',                 # Usamos MSE para regresión
            metrics=['mae']
        )
        
        print("X shape: ", input_shape[1:])
        print("y shape: ", output_shape)
        
        return model

    def train_model(self, X, y):
        """
        Método para entrenar el modelo.
        
        - X: Datos de entrada.
        - y: Datos de salida.
        """
        X = np.expand_dims(X.T, 0)
        y = np.expand_dims(y.T, 0)
        
        if self.model == None:
            raise Exception("El modelo no ha sido construido.")
        
        if X.shape[1:] != self.X:
            raise Exception("Los datos de entrada no tienen la forma correcta.")
        if y.shape[1:] != self.y:
            raise Exception("Los datos de salida no tienen la forma correcta.")

        # Entrenamiento
        self.model.fit(X, y, epochs=1, batch_size=32, validation_split=0)

    def evaluate_model(self, X_test, y_test):
        """
        Método para evaluar el modelo.
        
        - X_test: Datos de entrada de prueba.
        - y_test: Datos de salida de prueba.
        """
        if self.model == None:
            raise Exception("El modelo no ha sido construido.")
        
        loss, mae = self.model.evaluate(X_test, y_test)
        print(f"Loss: {loss}, MAE: {mae}")
        
    def predict(self, X):
        """
        Método para hacer predicciones con el modelo.
        
        - X: Datos de entrada.
        """
        if self.model == None:
            raise Exception("El modelo no ha sido construido.")
        
        predictions = self.model.predict(X)
        return predictions
