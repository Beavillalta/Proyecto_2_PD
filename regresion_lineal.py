# Parte 10 del proyecto

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Cargar el DataFrame con los datos procesados
df = pd.read_csv('datos_procesados.csv')

# Paso 1: Eliminar las columnas DEATH_EVENT, age y categoria_edad del dataframe para que sea la matriz X
X = df.drop(columns=['DEATH_EVENT', 'age', 'Grupo'])

# Paso 2: Ajustar una regresión lineal sobre el resto de columnas y usar la columna age como vector y
y = df['age']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar el modelo de regresión lineal
modelo_regresion = LinearRegression()

# Ajustar el modelo con los datos de entrenamiento
modelo_regresion.fit(X_train, y_train)

# Paso 3: Predecir las edades y comparar con las edades reales
y_pred = modelo_regresion.predict(X_test)

# Paso 4: Calcular el error cuadrático medio
error_cuadratico_medio = mean_squared_error(y_test, y_pred)

# Imprimir el error cuadrático medio
print(f"Error cuadrático medio: {error_cuadratico_medio}")

# También puedes imprimir las edades reales y predichas para comparar
print("\nEdades reales vs. Edades predichas:")
comparacion_edades = pd.DataFrame({'Edad Real': y_test, 'Edad Predicha': y_pred})
print(comparacion_edades)
