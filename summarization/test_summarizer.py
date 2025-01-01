import json
import random
import sys
import os
from summarizer import TextSummarizer

def get_random_article():
    """
    Lee un artículo aleatorio del dataset de arXiv
    """
    json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                            'bbdd_rag', 'arxiv-metadata-oai-snapshot.json')
    
    with open(json_path, 'r') as file:
        # Obtener el tamaño del archivo
        file.seek(0, 2)  # Ir al final del archivo
        file_size = file.tell()
        
        # Intentar hasta encontrar una línea válida
        while True:
            # Posición aleatoria en el archivo
            file.seek(random.randint(0, file_size))
            # Ir al inicio de la siguiente línea completa
            file.readline()
            
            # Leer la siguiente línea completa
            line = file.readline()
            if not line:  # Si llegamos al final, volver al principio
                file.seek(0)
                line = file.readline()
            
            try:
                paper = json.loads(line)
                return paper
            except json.JSONDecodeError:
                continue  # Si la línea no es válida, intentar otra vez

def main():
    # Inicializar el summarizer
    print("Inicializando summarizer...")
    summarizer = TextSummarizer()
    
    # Obtener un artículo aleatorio
    print("\nObteniendo artículo aleatorio...")
    paper = get_random_article()
    
    # Mostrar información del artículo
    print("\n" + "="*80)
    print(f"Título: {paper['title']}")
    print(f"ID: {paper['id']}")
    print(f"Categorías: {paper['categories']}")
    print("-"*80)
    print("Abstract original:")
    print(paper['abstract'])
    print("-"*80)
    
    # Generar y mostrar el resumen
    print("Resumen generado:")
    summary = summarizer.summarize(paper['abstract'])
    print(summary)
    print("="*80)

if __name__ == "__main__":
    main()
