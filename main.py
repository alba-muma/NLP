import os
import re
import pickle
import faiss
import torch
from bbdd_rag.create_vector_db import load_data, create_index
from bbdd_rag.search import SemanticSearch
from nlp_llm.load_model import generate_text, read_prompt, get_input_tokens
from language_translation.translation_utils import process_input, process_output
from summarization.summarizer import TextSummarizer
import pandas as pd

def main():
    # Check if the index exists; otherwise, create it
    if not os.path.exists('./bbdd_rag/arxiv_data.pkl'):
        print("Creando índice FAISS y base de datos vectorial...")
        df = pd.read_csv('./bbdd_rag/output.csv') if os.path.exists('./bbdd_rag/output.csv') else load_data(num_samples=50000)
        # Guardar datos
        with open('./bbdd_rag/arxiv_data.pkl', 'wb') as f:
            pickle.dump(df, f)

        print("Datos procesados y almacenados.")
    if not os.path.exists('./bbdd_rag/arxiv_index.faiss'):
        df = pd.read_csv('./bbdd_rag/output.csv')
        print("Creando índice FAISS...")
        index = create_index(df)
        print("Guardando índice...")
        faiss.write_index(index, './bbdd_rag/arxiv_index.faiss')

    # Inicializar sistema de búsqueda y summarizer
    searcher = SemanticSearch()
    summarizer = TextSummarizer()

    # Loop de consultas
    while True:
        query = input("\nIntroduce una consulta (o 'q' para salir): ").strip()
        if query.lower() == 'q':
            break
        
        # Detectar idioma y traducir query si es necesario
        cond_1 = len(query.strip()) >= 20
        cond_2 = len(re.findall(r'[a-zA-Z\u00C0-\u00FF]', query)) / len(query.strip()) >= 0.4
        if cond_1 and cond_2:
            query_en, original_lang = process_input(query)
            if original_lang and original_lang != 'en':
                print(f"\nIdioma detectado: {original_lang} (confianza ≥ 85%)")
                print(f"Traducción al inglés: {query_en}")
                query_for_search = query_en
            else:
                query_for_search = query
        else:
            if not cond_1:
                print("\nNota: El texto es demasiado corto para detectar el idioma de forma fiable. Se asume el inglés.")
            elif not cond_2:
                print("\nNota: El texto contiene muy pocas letras para detectar el idioma de forma fiable.")
            query_for_search = query

        # Detectar idioma y traducir query si es necesario
        query_en, original_lang = process_input(query)
        if original_lang and original_lang != 'en':
            print(f"\nIdioma detectado: {original_lang} (confianza ≥ 85%)")
            print(f"Traducción al inglés: {query_en}")
            query_for_search = query_en
        else:
            if len(query.strip()) < 20:
                print("\nNota: El texto es demasiado corto para detectar el idioma de forma fiable (mínimo 10 palabras). Se asume el inglés.")
            elif len(re.findall(r'[a-zA-Z\u00C0-\u00FF]', query)) / len(query.strip()) < 0.4:
                print("\nNota: El texto contiene muy pocas letras para detectar el idioma de forma fiable (mínimo 40% letras)")
            query_for_search = query

        # Buscar artículos relevantes
        print("\nBuscando artículos relevantes...")
        results = searcher.search(query_for_search)

        if not results:
            response = "I'm sorry, but I don't have specific papers on this topic. It seems your research may be novel. However, to ensure it, you can try expanding your search in other databases."
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
            print(f"Summary: {result['summary']}")
            print("-" * 80)

        # Crear prompt para el modelo
        papers_dict = {
            "papers": [{"title": r['title'], "summary": r['summary']} for r in results[0:2]]
        }

        # Vaciar la memoria de la GPU
        # Liberar memoria GPU
        del summarizer
        torch.cuda.empty_cache()
        
        # Leer el prompt
        prompt_base = read_prompt("./nlp_llm/prompts/prompt_0")
        
        # Generar el prompt completo
        user_query = f"{papers_dict}\nUser: {query_for_search}"
        full_prompt = (
            prompt_base + '\n' +
            "Papers: " + 
            user_query + '\n' +
            "Response: "
        )

        # print('-----------------------')
        # print(papers_dict)
        # print('-----------------------')
        print('num tokens:', get_input_tokens(full_prompt))
        generated = generate_text(max_length= get_input_tokens(full_prompt), prompt=full_prompt)
        print('-----------------------')
        print(generated)
        print('-----------------------')
        response = generated[generated.rfind('Summary:'):generated.rfind('<STOP>')]
        
        # Traducir respuesta si la query no estaba en inglés
        if original_lang and original_lang != 'en':
            response = process_output(response, original_lang)
        
        # print(response)

        torch.cuda.empty_cache()
        print(f"\n>>> {generated[generated.rfind('Response:') + len('Response: '):generated.rfind('<STOP>')]}")


if __name__ == "__main__":
    main()
