import os
import json
import pickle
from bbdd_rag.create_vector_db import load_data, create_index
from bbdd_rag.search import SemanticSearch
from nlp_llm.load_model import generate_text, read_prompt, get_input_tokens
import faiss
import torch
from language_translation.translation_utils import process_input, process_output
import re

def main():
    print("Sistema de búsqueda y generación de texto para artículos de arXiv\n")
    # Check if the index exists; otherwise, create it
    if not os.path.exists('./bbdd_rag/arxiv_index.faiss') or not os.path.exists('./bbdd_rag/arxiv_data.pkl'):
        print("Creando índice FAISS y base de datos vectorial...")
        df = load_data(num_samples=50000)  
        index = create_index(df)    
        df.to_csv('./bbdd_rag/aoutput.csv', index=False)

        print("Guardando índice y datos...")
        faiss.write_index(index, './bbdd_rag/arxiv_index.faiss')
        
        # Guardar datos
        with open('./bbdd_rag/arxiv_data.pkl', 'wb') as f:
            pickle.dump(df, f)

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
        
        # Detectar idioma y traducir query si es necesario
        query_en, original_lang = process_input(query)
        if original_lang and original_lang != 'en':
            print(f"\nIdioma detectado: {original_lang} (confianza ≥ 85%)")
            print(f"Traducción al inglés: {query_en}")
            query_for_search = query_en
        else:
            if len(query.strip()) < 10:
                print("\nNota: El texto es demasiado corto para detectar el idioma de forma fiable (mínimo 20 caracteres)")
            elif len(re.findall(r'[a-zA-Z\u00C0-\u00FF]', query)) / len(query.strip()) < 0.4:
                print("\nNota: El texto contiene muy pocas letras para detectar el idioma de forma fiable (mínimo 40% letras)")
            query_for_search = query

        # Buscar artículos relevantes
        print("\nBuscando artículos relevantes...")
        results = searcher.search(query_for_search)

        if not results:
            response = "I'm sorry, but I don't have specific papers in the current corpus on this topic. This does not imply that there are none. You can try expanding your search in other databases."
            # Traducir respuesta si la query no estaba en inglés
            if original_lang and original_lang != 'en':
                response = process_output(response, original_lang)
            print(response)
            continue

        # Mostrar resultados
        print("\nResultados más relevantes:")
        print("-" * 80)
        for result in results:
            print(f"{result['rank']}. {result['title']}")
            print(f"Score de similitud: {result['similarity_score']:.3f}")
            print(f"Categorías: {result['categories']}")
            print(f"ID: {result['id']}")
            # Mostrar solo los primeros 200 caracteres del abstract
            abstract = result['abstract'][:200]
            if len(result['abstract']) > 200:
                abstract += "..."
            print(f"Abstract: {abstract}")
            print("-" * 80)

        # Crear prompt para el modelo
        papers_dict = {
            "papers": [{"title": r['title'], "abstract": r['abstract']} for r in results]
        }

        # Vaciar la memoria de la GPU
        torch.cuda.empty_cache()
        
        # Leer el prompt
        prompt_base = read_prompt("./nlp_llm/prompts/prompt_0")
        
        # Generar el prompt completo
        user_query = f"{papers_dict}\nUser: {query_for_search}"
        full_prompt = (
            prompt_base + '\n' +
            "Papers: " + 
            user_query + '\n' +
            "Summary: "
        )

        # print('-----------------------')
        # print(papers_dict)
        # print('-----------------------')
        print('num tokens:', get_input_tokens(full_prompt))
        generated = generate_text(max_length= get_input_tokens(full_prompt), prompt=full_prompt)
        # print('-----------------------')
        # print(generated)
        # print('-----------------------')
        response = generated[generated.rfind('Summary:'):generated.rfind('<STOP>')]
        
        # Traducir respuesta si la query no estaba en inglés
        if original_lang and original_lang != 'en':
            response = process_output(response, original_lang)
        
        print(f"\n>>> {response}")

if __name__ == "__main__":
    print("Iniciando sistema...")
    main()
