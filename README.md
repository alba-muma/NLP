# ArXiv Search Engine

Este proyecto implementa un motor de búsqueda semántica para artículos científicos de arXiv utilizando embeddings y una base de datos vectorial FAISS.

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