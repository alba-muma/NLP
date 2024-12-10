# combined_system.py

from bbdd_rag.create_vector_db import load_data, create_index
from bbdd_rag.search import SemanticSearch
from nlp_llm.load_model import generate_text
import os
import json

def main():
    # Check if the index exists; otherwise, create it
    if not os.path.exists('./bbdd_rag/arxiv_index.faiss') or not os.path.exists('./bbdd_rag/arxiv_data.pkl'):
        print("Creando índice FAISS y datos...")
        df = load_data(num_samples=10000)  
        index = create_index(df)          
        print("Datos procesados y almacenados.")
    else:
        print("Índice y datos existentes encontrados.")

    # Inicializar sistema de búsqueda
    searcher = SemanticSearch()

    # Loop de consultas
    while True:
        query = input("\nIntroduce una consulta (o 'q' para salir): ").strip()
        if query.lower() == 'q':
            break

        # Buscar artículos relevantes
        print("\nBuscando artículos relevantes...")
        results = searcher.search(query, k=5)

        if not results:
            print("No se encontraron resultados.")
            continue

        # Mostrar resultados
        print("\nArtículos encontrados:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['title']}")
            print(f"Categorías: {result['categories']}")
            print(f"Abstract: {result['abstract'][:200]}...\n")

        # Crear prompt para el modelo
        papers_context = {
            "papers": [{"title": r['title'], "abstract": r['abstract']} for r in results]
        }
        prompt = f"Estos son los artículos más relevantes:\n{json.dumps(papers_context, indent=2)}\n\nUser: {query}\nAssistant:"

        # Generar texto
        print("\nGenerando respuesta...")
        response = generate_text(prompt, max_length=512)
        print("\nRespuesta del modelo:")
        print(response)

if __name__ == "__main__":
    main()
