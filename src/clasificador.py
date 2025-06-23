from sklearn.ensemble import RandomForestRegressor
from data_handler import read_telemetry_file, desnormalize_channel, derivate_channel, desnormalize_data
import numpy as np
import shap
from xgboost import XGBRegressor  # Importar XGBRegressor
import utils
import matplotlib.pyplot as plt
import random
import json
from utils import Track, LapType
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


mask = None
output_path = "out/"
track_path = "tracks/tsukuba.json"

k = 1
kk = []
with open(track_path) as json_file:
        track_data = json.load(json_file)
        track = Track(track_data["track"], track_data["length"]
                      , track_data["label_max"], track_data["sections"])

rng = np.random.default_rng(42)
random.seed(42)
resultados = []
resultados2 = []
modelos = []
x_series_s = []
score = 0
for modelo in [XGBRegressor(objective="reg:squarederror", n_estimators=100, max_depth=6, learning_rate=0.1), RandomForestRegressor()]:
    # Hardcopy de resultados en resultados2
    resultados2 = resultados.copy()
    resultados = []
    kk = []
    
    for k in range(len(track.sections)):
        X_series_train, X_series_test, X_scalar_train, X_scalar_test, y_train, y_test, minmax_label_u,  minmax_scalar_u, mask = read_telemetry_file(f"{output_path}datasets/dataset{k}.npz", 0, 100, 0.15, mask)

        new_X_series_train = []
        new_X_series_test = []

        for new in [new_X_series_train, new_X_series_test]:
            X_series = X_series_train if new is new_X_series_train else X_series_test
            for lap in range(len(X_series)):
                new.append([])  # Inicializar una lista para cada vuelta
                #lap_data = desnormalize_channel(X_series[lap])
                lap_data = X_series[lap]  # Usar los datos sin desnormalizar
                for channel in lap_data:
                    #channel = derivate_channel(channel)  # Derivar el canal
                    new[-1].append(np.mean(channel))

                # Añadir el tiempo acumulado como una característica adicional
                new[-1].append(X_scalar_train[lap] if new is new_X_series_train else X_scalar_test[lap])

        new_X_series_train = np.array(new_X_series_train)     
        new_X_series_test = np.array(new_X_series_test)

        #modelo = XGBRegressor(objective="reg:squarederror", n_estimators=100, max_depth=6, learning_rate=0.1)
        #modelo = RandomForestRegressor()

        print("Entrenando modelo para la sección:", track.sections[k].name)
        modelo.fit(new_X_series_train, y_train)  # X = variables de entrada, y = tiempo por vuelta
        predicciones = modelo.predict(new_X_series_test)
        predicciones1 = predicciones
        predicciones = np.abs(desnormalize_data(predicciones, minmax_label_u[0], minmax_label_u[1],     minmax_label_u[2]) - desnormalize_data(y_test, minmax_label_u[0], minmax_label_u[1], minmax_label_u[2]))
        resultados.append(predicciones)

        # print("Predicciones:", predicciones[:10])
        # print("Real:", y_test[:10])
        # print("Promedio de y:", np.mean(y_train))
        print("Valor promedio de error:", np.mean(np.abs(predicciones1 - y_test)))
        print("Valor promedio de error con la media:", np.mean(np.abs(y_test - np.mean(y_train))))
        score += (np.mean(np.abs(y_test - np.mean(y_train)))-np.mean(np.abs(predicciones1 - y_test)) / len  (track.sections)) 

        kk.append(np.mean(np.abs(desnormalize_data(y_test, minmax_label_u[0], minmax_label_u[1], minmax_label_u [2]) - np.mean(desnormalize_data(y_train, minmax_label_u[0], minmax_label_u[1], minmax_label_u[2])))))
        modelos.append(modelo)  # Guardar el modelo para cada sección
        x_series_s.append(new_X_series_train)

        # Desnormalizar predicciones y valores reales para calcular las métricas en segundos reales
        y_test_desnorm = desnormalize_data(y_test, minmax_label_u[0], minmax_label_u[1], minmax_label_u[2])
        y_pred_desnorm = desnormalize_data(predicciones1, minmax_label_u[0], minmax_label_u[1], minmax_label_u  [2])

        # Calcular métricas
        mae = mean_absolute_error(y_test_desnorm, y_pred_desnorm) / 1000
        mse = root_mean_squared_error(y_test_desnorm, y_pred_desnorm) / 1000
        r2 = r2_score(y_test_desnorm, y_pred_desnorm)

        # Imprimir resultados
        print(f"MAE (Error Absoluto Medio): \t\t\t{mae:3.0f} milisegundos")
        print(f"RMSE (Raíz Error Cuadrático Medio): \t\t{mse:3.0f} milisegundos")
        print(f"R² (Coeficiente de determinación): \t\t{r2:.3f}")

# Plotear los resultados
print("Score:", score)
k = 0
plt.figure(figsize=(10, 6))
resultados = np.array(resultados)
resultados = resultados
#plt.plot(resultados[:100], label=f"Lap {lap+1}")
resultados = np.mean(resultados, axis=1)
resultados2 = np.array(resultados2)
resultados2 = np.mean(resultados2, axis=1)
# Plot with a linea mas gruesa
plt.plot(np.arange(1, len(track.sections)+1), resultados/1000, linewidth=2, color="green", label="Random Forest MAE Predicciones")
plt.plot(np.arange(1, len(track.sections)+1), resultados2/1000, linewidth=2, color="blue", label="XGBoost MAE Predicciones")
# Plot average
plt.plot(np.arange(1, len(track.sections)+1), np.array(kk)/1000, linewidth=2, color="black", linestyle='--', label="MAE Media")

plt.xlabel("Sección")
plt.ylabel("Error  medio absoluto (ms)")
#plt.title("Predicciones de tiempo por vuelta")
plt.legend()
plt.savefig("out/classif/predicciones_tiempo_por_vuelta.png")
exit(0)

for modelo, new_X_series_train in zip(modelos, x_series_s):

    explainer = shap.Explainer(modelo, new_X_series_train)
    #explainer = shap.TreeExplainer(modelo)

    shap_values = explainer(new_X_series_train)
    
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': utils.model_channels_nobb,
        'mean_abs_shap': mean_abs_shap
    }).sort_values(by='mean_abs_shap', ascending=False)

    importance_df.to_csv(f"out/classif/csvs/importancia_shap{k}.csv", index=False)
    print(importance_df)
    print("")
    print("")

    shap.summary_plot(shap_values, new_X_series_train, feature_names=utils.model_channels_nobb)
    plt.savefig(f"out/classif/results/shap_summary_plot{k}.png")  # Guardar la gráfica como archivo
    print("Programa finalizado. Gráficas guardadas en 'out/'.")
    k+= 1