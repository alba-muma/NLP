import json
import pickle
import os
import torch
import sys
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
from joblib import Parallel, delayed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from summarization.summarizer import TextSummarizer

# Forzar el uso de CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(1)
summarizer = TextSummarizer()


def parallel_summarize(abstracts, num_gpus, batch_size):
    def summarize_batch(abstracts_batch):
        summaries = []
        for abstract in tqdm(abstracts_batch):
            summary = summarizer.summarize(abstract)
            summaries.append(summary)
        return summaries

    batches = [abstracts[i:i+batch_size] for i in range(0, len(abstracts), batch_size)]
    summaries = Parallel(n_jobs=num_gpus)(delayed(summarize_batch)(batch) for batch in batches)
    return [summary for batch in summaries for summary in batch]

def load_data(num_samples=1000):
    """
    Carga los datos del archivo JSON y los convierte en un DataFrame
    Args:
        num_samples: Número de muestras a cargar (None para cargar todo)
    """
    print("Cargando datos del archivo JSON...")
    with open('./bbdd_rag/arxiv-metadata-oai-snapshot.json', 'r') as file:
        data = []
        abstracts = []
        for i, line in enumerate(file):
            if num_samples is not None and i >= num_samples:
                break
            paper = json.loads(line)
            abstract = paper['abstract'].replace('\n', ' ')
            abstracts.append(abstract)
            data.append({
                'title': paper['title'].replace('\n', ' '),
                'abstract': abstract,
                'categories': paper['categories'],
                'id': paper['id']
            })

        # Procesar resúmenes en paralelo
        print("Generando resúmenes en paralelo...")
        batch_size = 50  # Ajusta según tu GPU
        num_gpus = 1 #torch.cuda.device_count()
        print(f"Usando {num_gpus} GPUs")
        
        summaries = parallel_summarize(abstracts, num_gpus=num_gpus, batch_size=batch_size)
        
        # Añadir resúmenes a los datos
        for i, summary in enumerate(summaries):
            data[i]['summary'] = summary

        print("Convirtiendo a DataFrame...")
        df = pd.DataFrame(data)
    
    return df

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
        faiss.write_index(index, './bbdd_rag/arxiv_index.faiss')
        
        # Guardar datos
        with open('./bbdd_rag/arxiv_data.pkl', 'wb') as f:
            pickle.dump(df, f)
        
        print("¡Proceso completado exitosamente!")
        print(f"Número de artículos procesados: {len(df)}")
        
    except Exception as e:
        print(f"Error durante el proceso: {str(e)}")

if __name__ == "__main__":
    main()
