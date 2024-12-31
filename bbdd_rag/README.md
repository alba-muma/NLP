# ArXiv Search Engine

Este módulo implementa un motor de búsqueda semántica para artículos científicos de arXiv utilizando embeddings y una base de datos vectorial FAISS.

## Descargar dataset
https://www.kaggle.com/datasets/Cornell-University/arxiv?resource=download 
El archivo se debe llamar "arxiv-metadata-oai-snapshot.json"

## Uso

1. Crear la base de datos vectorial:
```bash
python create_vector_db.py
```
Este proceso puede tardar varias horas dependiendo de tu hardware y el tamaño del dataset procesado. El resultado será guardado en `arxiv_index.faiss` y `arxiv_data.pkl`.

2. Realizar búsquedas:
```bash
python search.py
```

El script de búsqueda te permitirá introducir temas y encontrará los artículos más relevantes basándose en similitud semántica.

3. Visualizar la BBDD:
```bash
python view_bbdd.py
```

El script de visualización te permitirá explorar la base de datos vectorial.

## Implementación

- Utilizamos `sentence-transformers` (modelo all-MiniLM-L6-v2) para crear embeddings de los artículos
- FAISS para almacenamiento y búsqueda eficiente de vectores
- Los embeddings combinan título y abstract para capturar mejor el contenido del artículo
