import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv1D, GRU, Dense, Dropout, Concatenate, BatchNormalization, Lambda, LSTM, BatchNormalization, ReLU, LSTM, Dense, GlobalAveragePooling1D, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import MultiHeadAttention, Add, GlobalAveragePooling1D, SimpleRNN
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, RNN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber


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
            self.model = self.build_model()
        else:
            self.model = self.load_model(model)

    def build_model(self):
        """
        Método para construir y compilar el modelo.
        
        - X: Datos de entrada.
        - y: Datos de salida.
        """

        # Multi-head self-attention
        input_layer = Input(shape=(self.steps, self.channels))

        # Encoder stack (1 o 2 bloques según datos)
        x = self.transformer_encoder(input_layer, head_size=64, num_heads=32, ff_dim=1024, dropout=0)
        x = self.transformer_encoder(x, head_size=64, num_heads=32, ff_dim=1024, dropout=0)

        # Resumen global de la secuencia
        x = GlobalAveragePooling1D()(x)

        # Dense layers para regresión
        x = Dense(1024, activation='sigmoid')(x)
        #x = Dropout(0.1)(x)
        #x = Dropout(0.1)(x)
        output = Dense(1, activation='linear')(x)  # Salida para regresión

        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer='adam', loss=Huber(), metrics=['mae'])
        return model

    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0.1):
        # Multi-head self-attention
        x = LayerNormalization(epsilon=1e-6)(inputs)
        x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
        x = Add()([x, inputs])

        # Feed-forward
        x_skip = x
        x = LayerNormalization(epsilon=1e-6)(x)
        x = Dense(ff_dim, activation="relu")(x)
        x = Dropout(dropout)(x)
        x = Dense(inputs.shape[-1])(x)
        x = Add()([x, x_skip])
        return x
        
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
        self.model.fit(x=series_batch, y=label_batch, 
                       batch_size = batch, epochs=epochs, validation_split=training, shuffle=True)
        
    def predict(self, X):
        """
        Método para hacer predicciones con el modelo.
        
        - X: Datos de entrada.
        """
        if self.model == None:
            raise Exception("El modelo no ha sido construido.")
        
        X = np.transpose(X, (0, 2, 1))
        
        predictions = self.model.predict(X)
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
    
