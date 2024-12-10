# ArXiv Search Engine

Este proyecto implementa un motor de búsqueda semántica para artículos científicos de arXiv utilizando embeddings y una base de datos vectorial FAISS.

## Requisitos

1. Python 3.8+
2. Suficiente espacio en disco (el dataset es grande)

## Configuración
0. (Opcional) Crear un venv
``` bash
python -m venv .venv_rag
.\.venv_rag\Scripts\activate
```

1. Instalar dependencias:
```bash
pip install -r requirements.txt
```

2. Instalar dataset
https://www.kaggle.com/datasets/Cornell-University/arxiv?resource=download 
El archivo se debe llamar "arxiv-metadata-oai-snapshot.json"

## Uso

1. Crear la base de datos vectorial:
```bash
python create_vector_db.py
```
Este proceso puede tardar varias horas dependiendo de tu hardware y si procesas todo el dataset.

2. Realizar búsquedas:
```bash
python search.py
```

El script de búsqueda te permitirá introducir temas y encontrará los artículos más relevantes basándose en similitud semántica.

## Implementación

- Utilizamos `sentence-transformers` (modelo all-MiniLM-L6-v2) para crear embeddings de los artículos
- FAISS para almacenamiento y búsqueda eficiente de vectores
- Los embeddings combinan título y abstract para capturar mejor el contenido del artículo

## Notas

- La primera ejecución descargará automáticamente el dataset de Kaggle (~2GB)
- Los archivos generados (índice FAISS y datos) pueden ocupar bastante espacio
- Para pruebas, puedes modificar `num_samples` en `create_vector_db.py`
