import tensorflow as tf
from tensorflow.keras import layers, models

# Suponiendo que tus datos ya están cargados en X (inputs) y y (targets)
# X: shape (n_samples, n_features)
# y: shape (n_samples, n_outputs)

class Model():
    
    def __init__(self, X, y):
        """
        Constructor de la clase Model.
        
        - X: Datos de entrada.
        - y: Datos de salida.
        """
        self.X = X
        self.y = y
        self.build_model()

    def build_model(self):
        """
        Método para construir y compilar el modelo.
        
        - X: Datos de entrada.
        - y: Datos de salida.
        """
        # Definición del modelo
        input_shape = self.X.shape[1]
        output_shape = self.y.shape[1]

        model = models.Sequential([
            layers.Input(shape=(input_shape,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(output_shape)  # Última capa sin activación si es regresión
        ])

        model.compile(
            optimizer='adam',
            loss='mse',                 # Usamos MSE para regresión
            metrics=['mae']
        )

    def train_model(self, X, y):
        """
        Método para entrenar el modelo.
        
        - X: Datos de entrada.
        - y: Datos de salida.
        """
        if self.model == None:
            raise Exception("El modelo no ha sido construido.")

        # Entrenamiento
        self.model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)

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
