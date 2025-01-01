# Módulo de Language Translation

Este módulo implementa la funcionalidad de traducción automática utilizando modelos de Helsinki-NLP.

## Características

- Detección automática del idioma de origen
- Traducción automática de múltiples idiomas a inglés
- Traducción automática de inglés a múltiples idiomas

## Archivos

- `translation_utils.py`: Implementa funciones para detectar idiomas (mínimo 20 caracteres, 40% letras, confianza ≥85%) y traducir textos a/desde inglés usando modelos de Helsinki-NLP
- `translator.py`: Script para replicar el funcionamiento del traductor en el entorno real 

## Uso

```python
from translation_utils import process_input, process_output
# Detectar idioma y traducir si es necesario
process_input(text)
# Traducir de inglés a otro idioma
process_output(text, target_lang)
```

## Configuración

El módulo utiliza los modelos de Helsinki-NLP por defecto. Para un rendimiento óptimo los modelos se descargarán automáticamente en la primera ejecución

