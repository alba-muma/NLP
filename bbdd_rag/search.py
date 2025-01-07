import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import gc
import os
import spacy

# Forzar el uso de CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(1)

class SemanticSearch:
    def __init__(self):
        try:
            # Liberar memoria
            gc.collect()
            
            print("Cargando modelo de embeddings...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
            
            print("Cargando índice FAISS...")
            self.index = faiss.read_index('./bbdd_rag/arxiv_index.faiss')
            
            print("Cargando BBDD...")
            with open('./bbdd_rag/arxiv_data.pkl', 'rb') as f:
                self.df = pickle.load(f)
                
            self.nlp = spacy.load('en_core_web_md')
            
        except Exception as e:
            print(f"Error durante la inicialización: {str(e)}")
            raise
    
    def extract_keywords(self, text):
        """Extrae keywords usando spaCy"""
        doc = self.nlp(text.lower())
        # Obtener sustantivos y adjetivos relevantes
        keywords = [token.text for token in doc 
                   if (token.pos_ in ['NOUN', 'PROPN', 'ADJ']) 
                   and not token.is_stop]
        return ' '.join(keywords)

    def search(self, query):
        """
        Busca los k artículos más similares a la consulta
        """
        try:
            # Extraer keywords de la query
            query_keywords = self.extract_keywords(query)
            print(f"Keywords extraídos: {query_keywords}")
            
            # Usar keywords para la búsqueda si hay suficientes, sino usar query original
            search_text = query_keywords if len(query_keywords.split()) >= 2 else query
            
            # Crear embedding de la consulta usando CPU
            with torch.no_grad():
                query_vector = self.model.encode(search_text, convert_to_numpy=True)
            
            _vector = np.array([query_vector], dtype='float32')
            faiss.normalize_L2(_vector)
            
            # Buscar los k vecinos más cercanos
            k = self.index.ntotal
            distances, indices = self.index.search(_vector, k)
            # print(distances)
            
            # Obtener los resultados
            results = []
            alpha = 0.25  # puedes ajustar este valor según necesites
            
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                similarity = np.exp(-dist*(dist**2))
                # print(similarity)
                if similarity < 0.2:
                    break
                if idx >= len(self.df):
                    continue
                article = self.df.iloc[idx]
                results.append({
                    'rank': i + 1,
                    'similarity': similarity,
                    'distance': dist,
                    'title': article['title'],
                    'abstract': article['abstract'],
                    'summary': article['summary'],
                    'categories': article['main_topics'],
                    'id': article['id']
                })
            
            return results
        except Exception as e:
            print(f"Error durante la búsqueda: {str(e)}")
            return []

def main():
    try:
        print("Inicializando sistema de búsqueda...")
        searcher = SemanticSearch()
        
        while True:
            try:
                query = input("\nIntroduce un tema para buscar (o 'q' para salir): ").strip()
                if not query:
                    print("Por favor, introduce una consulta válida.")
                    continue
                if query.lower() == 'q':
                    break
                
                print("\nBuscando artículos relevantes...")
                results = searcher.search(query)
                
                if not results:
                    print("No se encontraron resultados.")
                    continue
                
                print("\nResultados más relevantes:")
                print("-" * 80)
                for result in results[0:2]:
                    print(f"\n{result['rank']}. {result['title']}")
                    print(f"Score de similitud: {result['similarity']:.3f}")
                    print(f"Distancia: {result['distance']:.3f}")
                    print(f"Categorías: {result['categories']}")
                    print(f"ID: {result['id']}")
                    print(f"Summary: {result['summary']}")
                    print("-" * 80)
            
            except KeyboardInterrupt:
                print("\nBúsqueda interrumpida por el usuario.")
                break
            except Exception as e:
                print(f"\nError durante la búsqueda: {str(e)}")
                continue
    
    except Exception as e:
        print(f"Error fatal: {str(e)}")
    finally:
        print("\n¡Gracias por usar el buscador semántico!")

if __name__ == "__main__":
    main()
