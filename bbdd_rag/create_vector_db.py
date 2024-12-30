import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
import pickle
import os
import torch

# Forzar el uso de CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(1)

def load_data(num_samples=1000):
    """
    Carga los datos del archivo JSON y los convierte en un DataFrame
    Args:
        num_samples: Número de muestras a cargar (None para cargar todo)
    """
    print("Cargando datos del archivo JSON...")
    with open('./arxiv-metadata-oai-snapshot.json', 'r') as file:
        data = []
        for i, line in enumerate(file):
            if num_samples is not None and i >= num_samples:
                break
            paper = json.loads(line)
            data.append({
                'title': paper['title'].replace('\n', ' '),
                'abstract': paper['abstract'].replace('\n', ' '),
                'categories': paper['categories'],
                'id': paper['id']
            })
    
    return pd.DataFrame(data)

def create_index(df):
    """
    Crea un índice FAISS para búsqueda eficiente
    Args:
        df: DataFrame con los datos
    """
    print("Inicializando modelo de embeddings...")
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    
    print("Generando embeddings para los artículos...")
    embeddings = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Combinar título y abstract para el embedding
        text = f"{row['title']} {row['abstract']}"
        with torch.no_grad():
            embedding = model.encode(text, convert_to_numpy=True)
        embeddings.append(embedding)
    
    # Convertir la lista de embeddings a un array de numpy
    embeddings = np.array(embeddings, dtype='float32')
    
    print("Creando índice FAISS...")
    # Crear índice
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    return index

def main():
    try:
        # Cargar datos
        df = load_data(num_samples=50000)
        # Crear índice
        index = create_index(df)
        df.to_csv('output.csv', index=False)
        
        print("Guardando índice y datos...")
        # Guardar índice
        faiss.write_index(index, 'arxiv_index.faiss')
        
        # Guardar datos
        with open('arxiv_data.pkl', 'wb') as f:
            pickle.dump(df, f)
        
        print("¡Proceso completado exitosamente!")
        print(f"Número de artículos procesados: {len(df)}")
        
    except Exception as e:
        print(f"Error durante el proceso: {str(e)}")

if __name__ == "__main__":
    main()
