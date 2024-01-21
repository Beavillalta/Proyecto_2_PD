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

#Parte_3 Calculando analíticas simples

# 1Verificar que los tipos de datos son correctos en cada colúmna (por ejemplo que no existan colúmnas numéricas en formato de cadena).

# Mostrar los tipos de datos actuales en cada columna
print("Tipos de datos antes de la verificación:")
print(df.dtypes)

# Verificar y corregir los tipos de datos
# Asegurarte de que las columnas numéricas están en formato numérico:
columnas_numericas = df.select_dtypes(include=['int', 'float']).columns
df[columnas_numericas] = df[columnas_numericas].apply(pd.to_numeric, errors='coerce')

# Mostrar los tipos de datos después de la verificación
print("\nTipos de datos después de la verificación:")
print(df.dtypes)

 #Calcular la cantidad de hombres fumadores vs mujeres fumadoras (usando agregaciones en Pandas).
cantidad_hombres_fumadores = df[(df['is_male'] == 1) & (df['is_smoker'] == 1)].shape[0]
cantidad_mujeres_fumadoras = df[(df['is_male'] == 0) & (df['is_smoker'] == 1)].shape[0]

print("Cantidad de hombres fumadores:", cantidad_hombres_fumadores)
print("Cantidad de mujeres fumadoras:", cantidad_mujeres_fumadoras)

#Parte 4 del Proyecto: Procesando información en bruto
#Realiza un GET request para descargarlos y escribe la respuesta como un archivo de texto plano con extensión csv (no necesitas pandas para esto, sólo manipulación de archivos nativa de Python)

import requests

def descargar_y_guardar_como_csv(url, nombre_archivo):
    try:
        # Realizar un GET request para descargar los datos
        respuesta = requests.get(url)
        respuesta.raise_for_status()  # Lanza una excepción si la respuesta tiene un código de estado diferente de 200

        # Guardar la respuesta en un archivo CSV
        with open(nombre_archivo, 'w', encoding='utf-8') as archivo:
            archivo.write(respuesta.text)

        print(f"Los datos fueron descargados exitosamente y guardados en {nombre_archivo}.")

    except requests.exceptions.RequestException as e:
        print(f"Error al realizar la solicitud: {e}")

# Ejemplo de uso
url_datos = "https://huggingface.co/datasets/mstz/heart_failure/raw/main/heart_failure_clinical_records_dataset.csv"
nombre_archivo_csv = "datos_descargados.csv"

descargar_y_guardar_como_csv(url_datos, nombre_archivo_csv)




