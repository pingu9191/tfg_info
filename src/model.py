import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from data_handler import read_telemetry_file, desnormalize_channel, derivate_channel
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.layers import Input, Conv1D, GRU, Dense, Dropout, Concatenate, BatchNormalization, Lambda, LSTM, BatchNormalization, ReLU, LSTM, Dense, GlobalAveragePooling1D, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import MultiHeadAttention, Add, GlobalAveragePooling1D, SimpleRNN
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, RNN, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras import layers, models


class MyModel():
    
    def __init__(self, channels, steps, model=None):
        """
        Constructor de la clase Model.
        
        - channels: Número de canales de entrada (dimensión de las series temporales).
        - steps: Número de pasos de tiempo (longitud de las series temporales).
        - model: Modelo de Keras construido y compilado.
        """
        
        self.channels = channels
        self.steps = steps
        
        if model is None:
            self.model = self.build_model_ii()
        else:
            self.model = self.load_model(model)

    """def build_model(self):
        
        #Método para construir y compilar el modelo.
        

        # Entrada
        input_layer = Input(shape=(self.steps, self.channels))

        # Bidirectional LSTM
        x = LSTM(64, return_sequences=False)(input_layer)
        #x = Bidirectional(LSTM(32, return_sequences=False))(x)
        #x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        output = Dense(1)(x)  # Salida de regresión

        # Construcción del modelo
        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss=['mse'], metrics=['mae'])

        return model"""
    def build_model_i(self):
        # Input 1: Telemetría secuencial (shape: steps x channels)
        sequence_input = Input(shape=(self.steps, self.channels), name='telemetry_input')

        # Input 2: Scalar (tiempo acumulado hasta ese tramo)
        scalar_input = Input(shape=(1,), name='offset_input')

        # Extracción de características con convoluciones
        x = Conv1D(filters=32, kernel_size=3, activation='relu')(sequence_input)
        x = BatchNormalization()(x)
        x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
        x = BatchNormalization()(x)

        # Modelado temporal con GRU
        x = GRU(100, return_sequences=True)(x)  # shape: (batch_size, 64)
        x = GRU(100, return_sequences=False)(x)  # shape: (batch_size, 64)
        x = Dropout(0.2)(x)

        # Fusionamos la representación de la secuencia con el scalar
        x = Concatenate()([x, scalar_input])  # shape: (batch_size, 101)

        # Capa densa que resume el tramo
        x = Dense(64, activation='relu')(x)  # shape: (batch_size, 128)
        x = Dropout(0.2)(x)

        # Capa final para predecir el tiempo de vuelta restante
        output = Dense(1, activation='relu')(x)

        # Definición del modelo con dos entradas
        model = Model(inputs=[sequence_input, scalar_input], outputs=output)

        # Compilación
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        return model
    
    def build_model_ii(self):
        """
        Método para construir y compilar el modelo.
        
        - X: Datos de entrada.
        - y: Datos de salida.
        """

        # Entrada
        input_layer = Input(shape=(self.steps, self.channels))
        scalar_input = Input(shape=(1,), name='offset_input')

        # Bidirectional LSTM
        x = Bidirectional(LSTM(64, return_sequences=True))(input_layer)
        x = Bidirectional(LSTM(128, return_sequences=False))(x)
        x = Dropout(0.2)(x)
        
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        x = Concatenate()([x, scalar_input])  # shape: (batch_size, 101)
        
        output = Dense(1, activation='relu')(x)  # Salida de regresión

        # Construcción del modelo
        model = Model(inputs=[input_layer, scalar_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss=Huber(), metrics=['mae'])

        return model

    def train_model(self, series_batch, scalar_batch, label_batch, batch=32, epochs=40, training=0.2):
        """
        Método para entrenar el modelo.
        
        - series_batch: Lote de datos de entrada de las series.
        - scalar_batch: Lote de datos escalares.
        - label_batch: Lote de datos de salida.
        """
        # Validación de los datos de entrada y salida
        series_batch = np.transpose(series_batch, (0, 2, 1))  # Change from (batch, channels, steps) to (batch, steps, channels)
        
        # Entrenamiento
        self.model.fit(x=[series_batch, scalar_batch], y=label_batch, 
                       batch_size = batch, epochs=epochs, validation_split=training, shuffle=True)
        
    def predict(self, X_series, X_scalar):
        """
        Método para hacer predicciones con el modelo.
        
        - X: Datos de entrada.
        """
        if self.model == None:
            raise Exception("El modelo no ha sido construido.")
        
        X_series = np.transpose(X_series, (0, 2, 1))
        
        predictions = self.model.predict([X_series, X_scalar])
        return predictions
    
    def evaluate_model(self, X_test, y_test):
        """
        Método para evaluar el modelo.
        
        - X_test: Datos de entrada de prueba.
        - y_test: Datos de salida de prueba.
        """
        if self.model == None:
            raise Exception("El modelo no ha sido construido.")
        
        X_test = np.transpose(X_test, (0, 2, 1))  # Change from (batch, channels, steps) to (batch, steps, channels)

        
        loss, mae = self.model.evaluate(X_test, y_test)
        print(f"Loss: {loss}, MAE: {mae}")
    
    def load_model(self, model_path):
        """
        Método para cargar un modelo guardado.
        
        - model_path: Ruta al archivo del modelo guardado.
        """
        return tf.keras.models.load_model(model_path)
