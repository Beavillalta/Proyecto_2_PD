 
import sys
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import requests
import numpy as np
import seaborn as sns  
from sklearn.manifold import TSNE
import plotly.graph_objects as go

df = pd.read_csv("datos_procesados.csv")
 

# Parte 9 del proyecto 

# Obtener el nombre de las columnas que no son numéricas
columnas_no_numericas = df.select_dtypes(exclude=['number']).columns

# Eliminar las columnas no numéricas del DataFrame
df = df.drop(columns=columnas_no_numericas)

# Paso 1: Eliminar la columna objetivo y convertir a array
X = df.drop(columns=['DEATH_EVENT', 'age']).values

# Paso 2: Exportar un array unidimensional de la columna objetivo
y = df['DEATH_EVENT'].values

# Paso 3: Ejecutar el algoritmo t-SNE
X_embedded = TSNE(
    n_components=3,
    learning_rate='auto',
    init='random',
    perplexity=3
).fit_transform(X)

# Paso 4: Crear un gráfico de dispersión 3D con Plotly
fig = go.Figure()

for label in set(y):
    indices = y == label
    scatter = go.Scatter3d(
        x=X_embedded[indices, 0],
        y=X_embedded[indices, 1],
        z=X_embedded[indices, 2],
        mode='markers',
        marker=dict(size=8, opacity=0.6),
        name=f'Clase {label}'
    )
    fig.add_trace(scatter)

# Configuraciones adicionales del diseño
fig.update_layout(scene=dict(
                    xaxis_title='Dimensión 1',
                    yaxis_title='Dimensión 2',
                    zaxis_title='Dimensión 3'),
                    width=800, height=800,
                    margin=dict(l=0, r=0, b=0, t=0))

# Guardar o mostrar la visualización
fig.write_html('scatter_3d_plotly.html')  # Puedes guardar el gráfico en un archivo HTML
# fig.show()  # O mostrarlo directamente en el entorno de ejecución