# Parte 7 del proyecto
import sys
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import requests
import numpy as np

df = pd.read_csv("datos_procesados.csv")

def graficar_histogramas(df):
    # Graficar la distribución de edades con un histograma
    plt.figure(figsize=(10, 6))
    plt.hist(df['age'], bins=20, color='skyblue', edgecolor='black', align='mid')
    plt.title('Distribución de Edades')
    plt.xlabel('Edad')
    plt.ylabel('Frecuencia')
    plt.grid(True)
    plt.show()

    # Graficar histogramas agrupados por hombre y mujer
    
     
    hombres = df[df['sex'] == 1]
    mujeres = df[df['sex'] == 0]

        # Calcular sumas para hombres
    smoking_m = hombres['smoking'].sum()
    diabetes_m = hombres['diabetes'].sum()
    anemia_m = hombres['anaemia'].sum()
    deaths_m = hombres['DEATH_EVENT'].sum()

        # Calcular sumas para mujeres
    smoking_f = mujeres['smoking'].sum()
    diabetes_f = mujeres['diabetes'].sum()
    anemia_f = mujeres['anaemia'].sum()
    deaths_f = mujeres['DEATH_EVENT'].sum()

        # Definir categorías y datos
    categorias = ['Anemicos', 'Diabeticos', 'Fumadores', 'Muertos']
    datos_m = [anemia_m, diabetes_m, smoking_m, deaths_m]
    datos_f = [anemia_f, diabetes_f, smoking_f, deaths_f]

    index = np.arange(len(datos_m))
    width = 0.30

        # Crear el gráfico
    fig, ax = plt.subplots()

    ax.bar(index-width/2, datos_m, width)
    ax.bar(index+width/2, datos_f, width)

    for i, j in zip(index, datos_m):
            ax.annotate(j, xy=(i-0.2,j+0.2))

    for i, j in zip(index, datos_f):
            ax.annotate(j, xy=(i+0.1,j+0.2))

    ax.set_title('Gráfico de comparación por género')
    ax.set_xticks(index)
    ax.set_xticklabels(categorias)
    ax.set_xlabel('Categorias')
    fig.legend(['Hombres', 'Mujeres'], loc='upper right', fontsize='small')

    fig.savefig('comparacion_por_genero_.png')
    plt.show()


graficar_histogramas(df)