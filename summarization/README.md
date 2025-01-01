# Módulo de Summarization

Este módulo implementa la funcionalidad de resumen de texto utilizando el modelo BART de Facebook.

## Características

- Resumen de textos utilizando BART (facebook/bart-large-cnn)
- Procesamiento por lotes para optimizar el uso de GPU
- Control de longitud del resumen basado en ratio del texto original
- Limpieza automática de memoria GPU
- Barra de progreso para monitorear el proceso

## Archivos

- `summarizer.py`: Implementa la clase TextSummarizer para generar resúmenes
- `test_summarizer.py`: Script para probar el summarizer con un artículo aleatorio

## Uso

```python
from summarizer import TextSummarizer

# Inicializar el summarizer
summarizer = TextSummarizer()

# Resumir un solo texto
summary = summarizer.summarize(text, ratio=0.5, min_length=20)

# Procesar múltiples textos en lotes
summaries = summarizer.process_batch(texts, batch_size=4, ratio=0.5, min_length=20)
```

## Configuración

El módulo detectará automáticamente si hay GPU disponible y ajustará el procesamiento en consecuencia. Para un rendimiento óptimo:

1. Asegúrate de tener CUDA instalado si usas GPU
2. Ajusta el `batch_size` según la memoria disponible
3. Usa el parámetro `ratio` para controlar la longitud del resumen
