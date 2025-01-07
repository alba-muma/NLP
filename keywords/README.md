# Módulo de Extracción de Keywords

Este módulo se encarga de la extracción automática de palabras clave (keywords) de documentos científicos, utilizando técnicas de procesamiento de lenguaje natural (NLP) y TF-IDF.
Se integra con el sistema principal de búsqueda semántica para extraer keywords relevantes del título y abstract de los artículos y mejorar la precisión de las búsquedas.

## Uso

```python
from keywords.keywords import generate_keywords

# Generar keywords para un DataFrame de documentos
df = generate_keywords(df, batch_size=1000)
```

## Implementación

- Procesamiento por lotes para manejar de forma eficiente grandes volúmenes de datos.