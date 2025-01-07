# 📚 Buscador de investigación académica

Sistema de búsqueda semántica para artículos científicos de arXiv que utiliza topic modeling, extracción de keywords, base de datos vectorial, embeddings, LLM y transformers para analizar la consulta de un usuario, encontrar y analizar artículos científicos relevantes para esta consulta y dar respuesta en el idioma de la consulta inicial del usuario.

## Componentes Principales

### Motor de Búsqueda (`search_engine.py`)

Implementa la lógica central del sistema:
- Búsqueda semántica usando embeddings y FAISS
- Integración con LLM para generar respuestas contextualizadas
- Procesamiento multilingüe de consultas
- Traducción de respuestas al idioma de usuario

### Interfaz de Usuario (`app.py`)

Implementa la interfaz web usando Streamlit:
- Búsqueda intuitiva con campo de texto
- Visualización de artículos en dos categorías:
  - Artículos Relevantes (similitud > 0.5)
  - Artículos de Interés (similitud ≤ 0.5)
- Respuesta del Sistema
- Indicador de tiempo de procesamiento

## Módulos de Soporte

- `bbdd_rag/`: Base de datos vectorial y búsqueda por similitud
- `keywords/`: Extracción de palabras clave
- `lda/`: Modelado de tópicos
- `llm_response/`: Generación de respuesta del Sistema con LLM
- `language_translation/`: Procesamiento multilingüe
- `summarization/`: Generación de resúmenes de los artículos

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