# Módulo de Generación de Texto con LLM

Este módulo gestiona la carga y uso del modelo de lenguaje Llama-3.2-1B para generar respuestas contextualizadas sobre artículos científicos.

Se integra con el sistema principal para generar resúmenes y análisis de los artículos encontrados, adaptando las respuestas al contexto de la consulta del usuario.

## Uso

```python
from nlp_llm.load_model import generate_text, read_prompt, get_input_tokens

# Leer un prompt predefinido
prompt = read_prompt('./nlp_llm/prompts/prompt_4')

# Generar texto
response = generate_text(prompt, max_length=2048)
```

## Implementación

- Utiliza el modelo Llama-3.2-1B
- Implementa cuantización de 4 bits para optimizar memoria
- Maneja prompts predefinidos para diferentes contextos
- Gestiona la generación de texto con parámetros configurables
