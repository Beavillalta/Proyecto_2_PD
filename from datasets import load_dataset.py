from datasets import load_dataset
import numpy as np

# Cargar el dataset
dataset = load_dataset("mstz/heart_failure")
data = dataset["train"]

# Obtener la lista de edades
edades = data["age"]

# Convertir la lista de edades a un arreglo de NumPy
edades_np = np.array(edades)

# Calcular el promedio de edad
promedio_edad = np.mean(edades_np)

# Imprimir el resultado
print("El promedio de edad de las personas participantes en el estudio es:", promedio_edad)

#Parte 2

import pandas as pd
#Convertir la estructura Dataset en un DataFrame de Pandas usando pd.DataFrame.
df = pd.DataFrame(data)

#Separar el dataframe en dos diferentes, uno conteniendo las filas con personas que perecieron (is_dead=1) y otro con el complemento.
fallecidos_df = df[df['is_dead'] == 1]
sobrevivientes_df = df[df['is_dead'] == 0]

#Calcular los promedios de las edades de cada dataset e imprimir.
promedio_edad_fallecidos = fallecidos_df['age'].mean()
promedio_edad_sobrevivientes = sobrevivientes_df['age'].mean()

#redondeo el resultado
edad_p_round_fallecidos = round(promedio_edad_fallecidos,1)
edad_p_round_sobrevivientes =round(promedio_edad_sobrevivientes,1)

#imprimimos resultados
print("Promedio de edad de los fallecidos:", edad_p_round_fallecidos)
print("Promedio de edad de los sobrevivientes:", edad_p_round_sobrevivientes)