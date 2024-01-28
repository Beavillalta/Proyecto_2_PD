 
import sys
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import requests
import numpy as np
import seaborn as sns  

df = pd.read_csv("datos_procesados.csv")
 

# Parte 8 del proyecto 

sns.set(style="whitegrid")
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

def plot_pie(ax, data, labels_map, title):
    values = data.value_counts()
    labels = [labels_map[x] for x in values.index]
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
    ax.set_title(title)

labels_map_anemicos = {0: 'No', 1: 'Si'}
plot_pie(axs[0, 0], df['anaemia'], labels_map_anemicos, 'Anemicos')

labels_map_diabetes = {0: 'No', 1: 'Si'}
plot_pie(axs[0, 1], df['diabetes'], labels_map_diabetes, 'Diabeticos')

labels_map_fumador = {0: 'No', 1: 'Si'}
plot_pie(axs[1, 0], df['smoking'], labels_map_fumador, 'Fumadores')

labels_map_muertos = {0: 'No', 1: 'Si'}
plot_pie(axs[1, 1], df['DEATH_EVENT'], labels_map_muertos, 'Fallecidos')

plt.subplots_adjust(wspace=0.4, hspace=0.4)
fig.suptitle('Diagramas de Torta por Característica', fontsize=16)

fig.savefig('graficos_tortas.png')
plt.show()