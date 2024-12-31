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
        self.summarizer = TextSummarizer()

    def process_query(self, query):
        # Detectar idioma y traducir si es necesario
        original_lang = None
        query_en = query
        
        # Procesar la consulta si no está en inglés
        if not re.match(r'^[a-zA-Z\s]*$', query):
            original_lang, query_en = process_input(query)
            query_for_search = query_en
        else:
            if len(query.strip()) < 10:
                print("\nNota: El texto es demasiado corto para detectar el idioma de forma fiable (mínimo 10 palabras). Se asume el inglés.")
            elif len(re.findall(r'[a-zA-Z\u00C0-\u00FF]', query)) / len(query.strip()) < 0.4:
                print("\nNota: El texto contiene muy pocas letras para detectar el idioma de forma fiable (mínimo 40% letras)")
            query_for_search = query
            
        # Realizar búsqueda semántica
        results = self.searcher.search(query_for_search)

        if not results:
            response = "I'm sorry, but I don't have specific papers on this topic. It seems your research may be novel. However, to ensure it, you can try expanding your search in other databases."
            # Traducir respuesta si la query no estaba en inglés
            if original_lang and original_lang != 'en':
                response = process_output(response, original_lang)
            return {"response": response, "papers": []}

        # Guardar los papers originales
        original_papers = results

        # Crear prompt para el modelo con resúmenes
        papers_dict = {
            "papers": [{"title": r['title'], "abstract": self.summarizer.summarize(r['abstract'])} for r in results[0:2]]
        }

        # Vaciar la memoria de la GPU
        del self.summarizer
        torch.cuda.empty_cache()
        
        # Leer el prompt
        prompt_base = read_prompt("./nlp_llm/prompts/prompt_1")
        
        # Generar el prompt completo
        user_query = f"{papers_dict}\nUser: {query_for_search}"
        full_prompt = prompt_base + user_query
        
        # Generar respuesta
        generated = generate_text(max_length=get_input_tokens(full_prompt), prompt=full_prompt)
        response = generated[generated.rfind('Summary:'):generated.rfind('<STOP>')]
        
        # Traducir respuesta si la query no estaba en inglés
        if original_lang and original_lang != 'en':
            response = process_output(response, original_lang)

        torch.cuda.empty_cache()
        
        # Devolver tanto la respuesta como los papers originales
        return {
            "response": response,
            "papers": original_papers
        }

    def __del__(self):
        # Limpiar memoria GPU al destruir la instancia
        torch.cuda.empty_cache()
