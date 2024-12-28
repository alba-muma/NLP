import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import gc
import os

# Forzar el uso de CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(1)

class SemanticSearch:
    def __init__(self):
        try:
            # Liberar memoria
            gc.collect()
            
            print("Cargando modelo...")
            # Usar un modelo más ligero y configurar explícitamente para CPU
            # self.model = SentenceTransformer('paraphrase-MiniLM-L3-v2', device='cuda')
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
            
            print("Cargando índice FAISS...")
            self.index = faiss.read_index('./bbdd_rag/arxiv_index.faiss')
            
            print("Cargando datos...")
            with open('./bbdd_rag/arxiv_data.pkl', 'rb') as f:
                self.df = pickle.load(f)
            
            print("Sistema inicializado correctamente!")
        except Exception as e:
            print(f"Error durante la inicialización: {str(e)}")
            raise
    
    def search(self, query, k=50):
        """
        Busca los k artículos más similares a la consulta
        """
        try:
            print("Procesando consulta...")
            # Crear embedding de la consulta usando CPU
            with torch.no_grad():
                query_vector = self.model.encode(query, convert_to_numpy=True)
            
            # Asegurar que el vector es float32 y tiene la forma correcta
            query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)
            query_norm = query_vector / np.linalg.norm(query_vector)
            
            print("Buscando coincidencias...")
            # Buscar los k vecinos más cercanos
            k = min(k, len(self.df))
            distances, indices = self.index.search(query_norm, k)
            print(distances)
            
            # Obtener los resultados
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                similarity = 1 / (1 + dist)
                if similarity < 0.4:
                    break
                if idx >= len(self.df):
                    continue
                article = self.df.iloc[idx]
                results.append({
                    'rank': i + 1,
                    'similarity_score': similarity,
                    'title': article['title'],
                    'abstract': article['abstract'],
                    'categories': article['categories'],
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
                for result in results:
                    print(f"\n{result['rank']}. {result['title']}")
                    print(f"Score de similitud: {result['similarity_score']:.3f}")
                    print(f"Categorías: {result['categories']}")
                    print(f"ID: {result['id']}")
                    # Mostrar solo los primeros 200 caracteres del abstract
                    abstract = result['abstract'][:200]
                    if len(result['abstract']) > 200:
                        abstract += "..."
                    print(f"Abstract: {abstract}")
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
