import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Conv1D, GRU, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model


class MyModel():
    
    def __init__(self, channels, steps):
        """
        Constructor de la clase Model.
        
        - channels: Número de canales de entrada (dimensión de las series temporales).
        - steps: Número de pasos de tiempo (longitud de las series temporales).
        - model: Modelo de Keras construido y compilado.
        """
        
        self.channels = channels
        self.steps = steps
        
        self.model = self.build_model()

    def build_model(self):
        """
        Método para construir y compilar el modelo.
        
        - X: Datos de entrada.
        - y: Datos de salida.
        """
        series_input = Input(shape=(self.channels, self.steps), name="series_input")
        scalar_input = Input(shape=(1,), name="scalar_input")
        
        # Capa de convolución
        x = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(series_input)
        x = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(x)
        x = Dropout(0.3)(x)
        
        # Capa GRU
        x = GRU(64, return_sequences=True)(x)
        x = GRU(64, return_sequences=False)(x)
        
        # Concatenación de la entrada escalar
        merged = Concatenate()([x, scalar_input])
        x = Dense(64, activation='relu')(merged)
        x = Dropout(0.3)(x)
        
        # Capa de salida
        output = Dense(1, activation='linear', name="output")(x)
        
        # Construcción del modelo
        model = Model(inputs=[series_input, scalar_input], outputs=output)
        model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['mae'])
        
        return model


    def train_model(self, series_batch, scalar_batch, label_batch):
        """
        Método para entrenar el modelo.
        
        - series_batch: Lote de datos de entrada de las series.
        - scalar_batch: Lote de datos escalares.
        - label_batch: Lote de datos de salida.
        """
        # Validación de los datos de entrada y salida
        

        # Entrenamiento
        self.model.train_on_batch([series_batch, scalar_batch], label_batch)


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
    
    def load_model(model_path):
        """
        Método para cargar un modelo guardado.
        
        - model_path: Ruta al archivo del modelo guardado.
        """
        return tf.keras.models.load_model(model_path)
    
