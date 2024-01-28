# Parte 11 del proyecto

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def graficar_distribucion_clases_1(df, columna_clases='DEATH_EVENT'):
    # Paso 1: Graficar la distribución de clases utilizando seaborn
    plt.figure(figsize=(6, 4))
    sns.countplot(x=columna_clases, data=df)
    plt.title('Distribución de Clases')
    plt.xlabel(columna_clases)
    plt.ylabel('Cantidad de Observaciones')
    plt.savefig('distribucion_clases.png')  # Guardar la figura como imagen
    plt.show()

def particion_estratificada_1(df, columna_clases='DEATH_EVENT'):
    # Paso 2: Realizar la partición estratificada del dataset
    X = df.drop(columns=[columna_clases])
    y = df[columna_clases]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def ajustar_arbol_decision_1(X_train, X_test, y_train, y_test):
    # Paso 3: Ajustar un árbol de decisión y calcular el accuracy
    # Paso 4: Experimentar con ajustes de parámetros para mejorar la precisión
    modelo_arbol = DecisionTreeClassifier(random_state=42)
    modelo_arbol.fit(X_train, y_train)
    y_pred = modelo_arbol.predict(X_test)
    precision = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo de árbol de decisión: {precision}")

    # Opcional: Imprimir la matriz de confusión
    from sklearn.metrics import confusion_matrix
    matriz_confusion = confusion_matrix(y_test, y_pred)
    print("Matriz de Confusión:")
    print(matriz_confusion)

import pandas as pd
 
# Cargar el DataFrame con los datos procesados
df = pd.read_csv('datos_procesados.csv')
df = df.drop(columns=['Grupo'])

# Utilizar las funciones de clasificación
graficar_distribucion_clases_1(df)
X_train, X_test, y_train, y_test = particion_estratificada_1(df)
ajustar_arbol_decision_1(X_train, X_test, y_train, y_test)