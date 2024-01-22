import sys
import pandas as pd
import requests
from io import StringIO

def descargar_datos_y_procesar(url):
    try:
        # Realizar un GET request para descargar los datos
        respuesta = requests.get(url)
        respuesta.raise_for_status()  # Lanza una excepción si la respuesta tiene un código de estado diferente de 200
        datos_csv = respuesta.text

        # Convertir los datos a un DataFrame
        dataframe = pd.read_csv(StringIO(datos_csv))

        # Categorizar en grupos  
        # Ejemplo: Crear una columna 'Grupo' basada en alguna condición
        dataframe['Grupo'] = pd.cut(dataframe['age'], bins=[0, 12, 19, 39, 59, float('inf')],
                                   labels=['Niño', 'Adolescente', 'Joven Adulto', 'Adulto', 'Adulto Mayor'],
                                   right=False)

        # Exportar el DataFrame a un archivo CSV resultante
        dataframe.to_csv('datos_procesados.csv', index=False)

        print("Operaciones completadas con éxito. Resultados guardados en datos_procesados.csv")

    except requests.exceptions.RequestException as e:
        print(f"Error al realizar la solicitud: {e}")

    except pd.errors.ParserError as e:
        print(f"Error al procesar el CSV: {e}")

if __name__ == "__main__":
    # Verificar si se proporcionó al menos un argumento (la URL)
    if len(sys.argv) < 2:
        print("Por favor, proporcione la URL como argumento.")
        sys.exit(1)

    # Obtener la URL desde los argumentos de la línea de comandos
    url = sys.argv[1]

    # Llamar a la función para descargar datos y procesar
    descargar_datos_y_procesar(url)

