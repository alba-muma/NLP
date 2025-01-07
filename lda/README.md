# Módulo de Modelado de Tópicos

Este módulo implementa el modelado de tópicos usando LDA (Latent Dirichlet Allocation) para descubrir automáticamente los temas principales en la colección de artículos científicos.

Se integra con el sistema principal para categorizar los artículos y mejorar la relevancia de las búsquedas mediante la identificación de temas comunes.

## Uso

```python
from lda.topic_modeling import perform_topic_modeling

# Realizar modelado de tópicos en el DataFrame
df = perform_topic_modeling(df)
```

## Implementación

- Utiliza Gensim para el modelado LDA
- Procesa documentos en lotes para eficiencia
- Asigna múltiples tópicos a cada documento con sus probabilidades
- Almacena los tópicos procesados para su uso posterior
