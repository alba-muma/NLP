import os
import re
import pickle
import faiss
import torch
import numpy as np
from bbdd_rag.create_vector_db import load_data, create_index
from bbdd_rag.search import SemanticSearch
from nlp_llm.load_model import generate_text, read_prompt, get_input_tokens
from language_translation.translation_utils import process_input, process_output

class SearchEngine:
    def __init__(self):
        # Check if the index exists; otherwise, create it
        if not os.path.exists('./bbdd_rag/arxiv_index.faiss') or not os.path.exists('./bbdd_rag/arxiv_data.pkl'):
            print("Creando índice FAISS y base de datos vectorial...")
            df = load_data()
            index = create_index(df)
            
            # Save the index and data
            faiss.write_index(index, './bbdd_rag/arxiv_index.faiss')
            with open('./bbdd_rag/arxiv_data.pkl', 'wb') as f:
                pickle.dump(df, f)
        
        # Inicializar sistema de búsqueda y summarizer
        self.searcher = SemanticSearch()

    def process_query(self, query):
        # Detectar idioma y traducir si es necesario
        original_lang = None
        query_en = query
        language_info = {}
        
        # Procesar la consulta
        cond_1 = len(query.strip()) >= 20
        cond_2 = len(re.findall(r'[a-zA-Z\u00C0-\u00FF]', query)) / len(query.strip()) >= 0.4
        # print(f"cond_1: {cond_1}, cond_2: {cond_2}")
        if cond_1 and cond_2:
            query_en, original_lang = process_input(query)
            if original_lang and original_lang != 'en':
                query_for_search = query_en
                language_info = {
                    "detected": True,
                    "lang": original_lang,
                    "translated_query": query_en
                }
            else:
                language_info = {
                    "detected": True,
                    "lang": 'en',
                    "translated_query": query
                }
                query_for_search = query
        else:
            if not cond_1:
                # print('cond 1')
                language_info = {
                    "detected": False,
                    "warning": "El texto es demasiado corto para detectar el idioma de forma fiable. Se asume el inglés."
                }
            elif not cond_2:
                # print('cond 2')
                language_info = {
                    "detected": False,
                    "warning": "El texto contiene muy pocas letras para detectar el idioma de forma fiable."
                }
            query_for_search = query
        
        # Realizar búsqueda semántica
        results = self.searcher.search(query_for_search)

        if not results:
            response = "I'm sorry, but I don't have specific papers on this topic. It seems your research may be novel. However, to ensure it, you can try expanding your search in other databases."
            # Traducir respuesta si la query no estaba en inglés
            if original_lang and original_lang != 'en':
                response = process_output(response, original_lang)
            return {"response": response, "papers": [], "language_info": language_info}

        # Guardar los papers originales con sus scores
        original_papers = []
        for r in results:
            paper_with_score = {
                'title': r['title'],
                'abstract': r['abstract'],
                'summary': r['summary'],
                'similarity': r['similarity'],
                'categories': r['categories']
            }
            original_papers.append(paper_with_score)

        # Crear prompt para el modelo con resúmenes
        papers_dict = {
            "papers": [{"title": r['title'].replace('\n', ' '), "summary": r['summary'].replace('\n', ' ')} for r in results[0:2] if r['similarity']>0.7]
        }

        # Vaciar la memoria de la GPU
        torch.cuda.empty_cache()
        # Lecer el prompt correspondiente
        prompt = 4
        if not papers_dict["papers"]:
            print('NO HAY ARTÍCULOS MUY RELEVANTES')
            papers_dict = {
            "papers": [{"title": r['title'].replace('\n', ' '), "summary": r['summary'].replace('\n', ' ')} for r in results[0:2]]
            }
            prompt = 5
        prompt_base = read_prompt(f"./nlp_llm/prompts/prompt_{prompt}")
        
        # Generar el prompt completo
        user_query = f"{papers_dict}\nUser: {query_for_search}"
        # print('user_query:', user_query)
        full_prompt = prompt_base + '\n' + user_query + '\n' + "Response:"
        
        # Generar respuesta
        generated = generate_text(max_length=get_input_tokens(full_prompt), prompt=full_prompt)
        print('generated:', generated)
        response = generated.split('Response:')[prompt].split('<STOP>')[0]
       
        # Traducir respuesta si la query no estaba en inglés
        if language_info.get("detected", False):
            response = process_output(response, language_info["lang"])

        torch.cuda.empty_cache()
        
        # Devolver tanto la respuesta como los papers originales
        return {
            "response": response,
            "papers": original_papers,
            "language_info": language_info
        }

    def __del__(self):
        # Limpiar memoria GPU al destruir la instancia
        torch.cuda.empty_cache()
