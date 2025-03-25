import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo CSV
data = pd.read_csv('telemetria_breve.csv', header=10, delimiter=';', nrows=40000, low_memory=False)

# Limpiar los nombres de las columnas de espacios adicionales
data.columns = data.columns.str.strip()

# Convertir las columnas 'throttle' y 'LapDist' a numérico, reemplazando comas si es necesario
if 'Throttle' in data.columns and 'LapDist' in data.columns:
    # Reemplazar comas solo si los datos son de tipo texto
    if data['Throttle'].dtype == 'object':
        data['Throttle'] = pd.to_numeric(data['Throttle'].str.replace(',', '.'), errors='coerce')
    else:
        data['Throttle'] = pd.to_numeric(data['Throttle'], errors='coerce')

    if data['LapDist'].dtype == 'object':
        data['LapDist'] = pd.to_numeric(data['LapDist'].str.replace(',', '.'), errors='coerce')
    else:
        data['LapDist'] = pd.to_numeric(data['LapDist'], errors='coerce')

    # Extraer los datos
    throttle = data['Throttle']
    lapdist = data['LapDist']

    # Crear el gráfico de la columna 'throttle'
    plt.figure(figsize=(10, 5))
    plt.plot(throttle, label='Throttle')
    plt.xlabel('Tiempo (o índice de muestra)')
    plt.ylabel('Throttle (%)')
    plt.title('Gráfico de Throttle con Marcadores de Inicio de Vuelta')

    # Colocar rayas verticales en cada punto donde 'LapDist' sea 0
    lap_start_indices = data.index[data['LapDist'] == 0].tolist()
    for idx in lap_start_indices:
        plt.axvline(x=idx, color='red', linestyle='--', linewidth=0.7, label='Inicio de Vuelta' if idx == lap_start_indices[0] else "")

    plt.legend()
    plt.show()
else:
    print("Las columnas 'throttle' y 'LapDist' no están en el archivo CSV.")
