# Módulo de LDA (Latent Dirichlet Allocation)

Este módulo implementa el análisis de tópicos utilizando LDA (Latent Dirichlet Allocation).

## Características

- Topic_Modelling incluye el notebook para hacer el LDA, sin la parte de añadir los main_topics a cada paper, pero permite hacer grafos en Colab.
- Topic_Modelling_sin_grafo incluye el notebook para hacer LDA y los main_topics de cada paper, no permite hacer grafos en Colab porque se alcanza el límite de RAM.

El archivo final que incluye los main topics para cada paper se llama **nuevo_dataframe_80_keywords.csv**. El archivo **nuevo_dataframe_80.csv** contiene el conjunto de datos original con títulos+abstracts preprocesados y la distribución de tópicos del LDA. El archivo **keywords_topics** incluye los nombres clave para cada tópico y se usa junto con el anterior para crear el archivo final (nuevo_dataframe_80_keywords.csv). El archivo processed_texts.txt se utiliza al inicio del notebook para cargar los datos ya preprocesados y poder hacer a partir de él todo el procesado.