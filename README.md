# 📚 Buscador de investigación académica

Este proyecto implementa un motor de búsqueda semántica para artículos científicos de arXiv utilizando embeddings y una base de datos vectorial FAISS. Se proporciona una interfaz de usuario para realizar búsquedas y visualizar la base de datos vectorial. El sistema está preparado para procesar consultas en cualquier idioma y responder en el idioma de la consulta. El resultado de la búsqueda muestra:
- Artículos relevantes basados en similitud semántica y su medida de similitud.
- Diferencias entre la línea de investigación del usuario y los artículos relevantes.
- Resumen de los artículos.
- Visualización de tópicos o categorías asociadas a los artículos.

## Requisitos

1. Python 3.10+
2. Suficiente espacio en disco (el dataset es grande)

## Configuración

### Paso 1: Crear el entorno virtual

1. Crear el entorno virtual:

    ```bash
    python -m venv myenv
    ```

2. Activar el entorno virtual:

    - En Windows:

        ```bash
        .\myenv\Scripts\activate
        ```

    - En macOS/Linux:

        ```bash
        source myenv/bin/activate
        ```

### Paso 2: Instalar las dependencias

Instalar las dependencias necesarias:

```bash
pip install -r requirements.txt
```

### Paso 3: Ejecutar la app principal
```bash
python -m streamlit run app.py
```