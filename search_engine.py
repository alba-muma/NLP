from sentence_transformers import SentenceTransformer, util
from nlp_llm.load_model import generate_text, read_prompt, get_input_tokens
import faiss
import torch
from language_translation.translation_utils import process_input, process_output
import re
import pickle
import numpy as np
import gc
import os

class SearchEngine:
    def __init__(self):
        print("Iniciando sistema...")
        # Cargar el modelo SBERT en GPU
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
        
        # Cargar datos desde el archivo pickle
        with open("./bbdd_rag/arxiv_data.pkl", 'rb') as f:
            data = pickle.load(f)
            self.titles = data['title'].replace('\n', ' ')
            self.abstracts = data['abstract'].replace('\n', ' ')
        
        # Ruta para los embeddings pre-calculados
        self.embeddings_file = "./bbdd_rag/title_embeddings.pt"
        
        # Cargar o generar embeddings de títulos
        if os.path.exists(self.embeddings_file):
            print("Cargando embeddings pre-calculados...")
            self.title_embeddings = torch.load(self.embeddings_file, map_location='cpu')
        else:
            print("Generando embeddings de títulos por primera vez...")
            with torch.cuda.device(0):
                self.title_embeddings = self.embedder.encode(self.titles.tolist(), convert_to_tensor=True)
                self.title_embeddings = self.title_embeddings.cpu()
            print("Guardando embeddings para uso futuro...")
            torch.save(self.title_embeddings, self.embeddings_file)
        
        # Convertir embeddings a numpy para FAISS
        embeddings_np = self.title_embeddings.numpy()
        
        # Inicializar índice FAISS
        print("Inicializando índice FAISS...")
        embedding_size = embeddings_np.shape[1]
        self.index = faiss.IndexFlatL2(embedding_size)
        self.index.add(embeddings_np)  # Añadir embeddings al índice
        
        self.prompt_base = read_prompt("./nlp_llm/prompts/prompt_0")
        print("Sistema iniciado correctamente!")

    def clear_gpu_memory(self):
        """
        Limpia la memoria CUDA/GPU
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def compute_similarities_batched(self, query_embedding, candidate_indices, batch_size=10):
        """
        Calcula similitudes en lotes pequeños para evitar problemas de memoria
        """
        all_scores = []
        query_embedding = query_embedding.to(self.device)
        
        for i in range(0, len(candidate_indices), batch_size):
            batch_indices = candidate_indices[i:i + batch_size]
            # Mover lote a GPU, calcular similitud y volver a CPU
            batch_embeddings = self.title_embeddings[batch_indices].to(self.device)
            batch_scores = util.pytorch_cos_sim(query_embedding, batch_embeddings)[0]
            all_scores.append(batch_scores.cpu().numpy())
            
            # Limpiar memoria GPU después de cada lote
            del batch_embeddings
            torch.cuda.empty_cache()
            
        return np.concatenate(all_scores)

    def normalize_scores(self, scores):
        """
        Normaliza los scores de similitud a un rango [0, 1]
        usando una función sigmoide modificada para dar más
        peso a las similitudes altas
        """
        # Aplicar una transformación sigmoide modificada
        normalized = 1 / (1 + np.exp((scores - 0.5)))
        
        # Re-escalar al rango [0, 1]
        if len(normalized) > 1:
            normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min())
        
        return normalized

    def process_query(self, query):
        """
        Process a query and return the system's response and additional information
        """
        try:
            # Detectar idioma y traducir query si es necesario
            query_en, original_lang = process_input(query)
            language_info = {}
            
            if original_lang and original_lang != 'en':
                language_info = {
                    "detected_lang": original_lang,
                    "translated_query": query_en
                }
                query_for_search = query_en
            else:
                if len(query.strip()) < 20:
                    language_info = {
                        "warning": "El texto es demasiado corto para detectar el idioma de forma fiable (mínimo 20 caracteres)"
                    }
                elif len(re.findall(r'[a-zA-Z\u00C0-\u00FF]', query)) / len(query.strip()) < 0.4:
                    language_info = {
                        "warning": "El texto contiene muy pocas letras para detectar el idioma de forma fiable (mínimo 40% letras)"
                    }
                query_for_search = query

            # Buscar artículos relevantes
            results = self.search(query_for_search)
            
            if not results:
                response = "I'm sorry, but I don't have specific papers in the current corpus on this topic. This does not imply that there are none. You can try expanding your search in other databases."
                if original_lang and original_lang != 'en':
                    response = process_output(response, original_lang)
                return {
                    "response": response,
                    "language_info": language_info,
                    "papers": []
                }

            # Preparar información de los papers encontrados
            papers = []
            papers_dict = ""
            for i, (score, idx) in enumerate(results, 1):
                title = self.titles[idx]
                abstract = self.abstracts[idx]
                papers.append({
                    "title": title,
                    "abstract": abstract,
                    "score": float(score)
                })
                papers_dict += f"[{i}] Title: {title}\nAbstract: {abstract}\n\n"

            # Generar el prompt completo
            user_query = f"{papers_dict}\nUser: {query_for_search}"
            full_prompt = (
                self.prompt_base + '\n' +
                "Papers: " + 
                user_query + '\n' +
                "Summary:"
            )
            print(full_prompt)

            # Generar respuesta
            generated = generate_text(max_length=get_input_tokens(full_prompt), prompt=full_prompt)
            response = generated[generated.rfind('Summary:') + len('Summary:'):generated.rfind('<STOP>')]
            
            # Traducir respuesta si la query no estaba en inglés
            if original_lang and original_lang != 'en':
                response = process_output(response, original_lang)

            return {
                "response": response,
                "language_info": language_info,
                "papers": papers
            }
            
        finally:
            # Limpiar memoria GPU después de cada consulta
            self.clear_gpu_memory()

    def search(self, query):
        """
        Búsqueda usando FAISS con gestión cuidadosa de memoria GPU
        """
        # Generar embedding de la consulta en GPU y mover inmediatamente a CPU
        with torch.cuda.device(0):
            query_embedding = self.embedder.encode(query, convert_to_tensor=True)
            query_embedding = query_embedding.cpu().numpy().reshape(1, -1)
            # Limpiar cualquier tensor temporal de la GPU
            torch.cuda.empty_cache()
        
        # Búsqueda con FAISS (en CPU)
        k = 100
        distances, indices = self.index.search(query_embedding, k)
        
        # Normalizar distancias a scores
        max_dist = np.max(distances)
        if max_dist > 0:
            scores = 1 - (distances / max_dist)
        else:
            scores = distances
        
        # Filtrar por umbral
        mask = scores[0] > 0.5
        filtered_scores = scores[0][mask]
        filtered_indices = indices[0][mask]
        
        if len(filtered_scores) == 0:
            return []
        
        # Ordenar por relevancia
        sorted_idx = np.argsort(-filtered_scores)
        filtered_scores = filtered_scores[sorted_idx]
        filtered_indices = filtered_indices[sorted_idx]
        
        # Devolver resultados
        results = [(float(score), int(idx)) for score, idx in zip(filtered_scores, filtered_indices)]
        return results
