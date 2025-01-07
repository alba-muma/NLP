import os
import re
import pickle
import faiss
from bbdd_rag.create_vector_db import load_data, create_index
from bbdd_rag.search import SemanticSearch
from llm_response.generate_response import generate_text, read_prompt, get_input_tokens
from language_translation.translation_utils import process_input, process_output

class SearchEngine:
    """
    Motor de búsqueda semántica que combina embeddings, búsqueda por similitud
    y generación de respuestas contextualizadas.
    """
    
    def __init__(self):
        """
        Inicializa el motor de búsqueda cargando el modelo de embeddings,
        el índice FAISS y los datos de los artículos.
        """
        # Check if the index exists; otherwise, create it
        if not os.path.exists('./bbdd_rag/arxiv_index.faiss') or not os.path.exists('./bbdd_rag/arxiv_data.pkl'):
            print("Creando índice FAISS y base de datos vectorial...")
            df = load_data()
            index = create_index(df)
            
            # Save the index and data
            faiss.write_index(index, './bbdd_rag/arxiv_index.faiss')
            with open('./bbdd_rag/arxiv_data.pkl', 'wb') as f:
                pickle.dump(df, f)
        
        # Inicializar sistema de búsqueda
        self.searcher = SemanticSearch()

    def process_query(self, query):
        """
        Procesa una consulta del usuario y genera una respuesta contextualizada.
        
        Args:
            query (str): Consulta del usuario
        
        Returns:
            dict: Resultados incluyendo artículos relevantes y respuesta generada
        """
        # Detectar idioma y traducir si es necesario
        original_lang = None
        language_info = {}
        
        # Procesar la consulta
        cond_1 = len(query.strip()) >= 20
        cond_2 = len(re.findall(r'[a-zA-Z\u00C0-\u00FF]', query)) / len(query.strip()) >= 0.4
        if cond_1 and cond_2:
            print('1')
            query_en, original_lang = process_input(query)
            if original_lang and original_lang != 'en':
                query_for_search = query_en
                language_info = {
                    "detected": True,
                    "lang": original_lang,
                    "translated_query": query_en
                }
            else:
                print('1.1')
                language_info = {
                    "detected": True,
                    "lang": 'en',
                    "translated_query": query
                }
                query_for_search = query
        else:
            print('2')
            if not cond_1:
                language_info = {
                    "detected": False,
                    "warning": "El texto es demasiado corto para detectar el idioma de forma fiable. Se asume el inglés."
                }
            elif not cond_2:
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
            "papers": [{"title": r['title'].replace('\n', ' '), "summary": r['summary'].replace('\n', ' ')} for r in results[0:2] if r['similarity']>0.5]
        }

        # Leer el prompt correspondiente
        prompt = 4
        if not papers_dict["papers"]:
            response = "I'm sorry, but I don't have relevant papers on this topic. Here is a list of some papers that may be similar to your research interest. Remember, you can try expanding your search in other databases."
            # Traducir respuesta si la query no estaba en inglés
            if original_lang and original_lang != 'en':
                response = process_output(response, original_lang)
        elif len(papers_dict["papers"]) == 1:
            response = f"{papers_dict["papers"][0]["title"]} is relevant to your research interest. Remember, you can try expanding your search in other databases."
            # Traducir respuesta si la query no estaba en inglés
            if original_lang and original_lang != 'en':
                response = process_output(response, original_lang)
        else:
            prompt_base = read_prompt(f"./llm_response/prompts/prompt_{prompt}")
        
            # Generar el prompt completo
            user_query = f"{papers_dict}\nUser: {query_for_search}"
            full_prompt = prompt_base + '\n' + user_query + '\n' + "Response:"
            
            # Generar respuesta
            response = generate_text(max_length=get_input_tokens(full_prompt), prompt=full_prompt)
        
            # Traducir respuesta si la query no estaba en inglés
            if language_info.get("detected", False):
                response = process_output(response, language_info["lang"])
        
        # Devolver tanto la respuesta como los papers originales
        return {
            "response": response,
            "papers": original_papers,
            "language_info": language_info
        }
