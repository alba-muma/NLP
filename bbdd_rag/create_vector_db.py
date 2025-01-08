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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from summarization.summarizer import TextSummarizer
from keywords.keywords import generate_keywords
from lda.topic_modeling import perform_topic_modeling

def load_data(num_samples=5000):
    """
    Carga los datos del archivo JSON y los convierte en un DataFrame
    Args:
        num_samples: Número de muestras a cargar (None para cargar todo)
    """
    # print(f"Cargando {num_samples if num_samples else 'todos los'} artículos del archivo JSON...")
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
        # print("Generando resúmenes en paralelo...")
        batch_size = 4  # Tamaño de lote reducido para evitar problemas de memoria
        
        # Inicializar summarizer
        summarizer = TextSummarizer()
        
        # Procesar todos los abstracts en lotes
        summaries = summarizer.process_batch(
            abstracts,
            batch_size=batch_size,
            ratio=0.5,
            min_length=20,
            show_progress=True
        )
        
        # Añadir resúmenes a los datos
        for i, summary in enumerate(summaries):
            data[i]['summary'] = summary
        
        # print("Convirtiendo a DataFrame...")
        df = pd.DataFrame(data)
        
        # print("Generando keywords...")
        df = generate_keywords(df)

        # print("Generando categorías...")
        df = perform_topic_modeling(df)

        df.to_csv('./bbdd_rag/output.csv', index=False)
    
    return df

def create_index(df):
    """
    Crea un índice FAISS para búsqueda eficiente
    Args:
        df: DataFrame con los datos
    """
    # print("Inicializando modelo de embeddings...")
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    
    # print("Generando embeddings para los artículos...")
    embeddings = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Combinar los elementos de la lista 'keywords' en una cadena separada por espacios
        keywords_list = eval(row['keywords']) if isinstance(row['keywords'], str) else row['keywords']
        text = " ".join(keywords_list)
        
        with torch.no_grad():
            embedding = model.encode(text, convert_to_numpy=True)
        embeddings.append(embedding)

    # Convertir la lista de embeddings a un array de numpy
    embeddings = np.array(embeddings, dtype='float32')
    
    # print("Creando índice FAISS...")
    # Crear índice
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    return index

def main():
    try:
        if not os.path.exists('./bbdd_rag/arxiv_data.pkl'):
            # Cargar datos
            df = load_data(num_samples=50000)
            # Guardar datos en pickle y csv
            with open('./bbdd_rag/arxiv_data.pkl', 'wb') as f:
                pickle.dump(df, f)
            df.to_csv('./bbdd_rag/output.csv', index=False)
        else:
            # Cargar datos desde el archivo
            with open('./bbdd_rag/arxiv_data.pkl', 'rb') as f:
                df = pickle.load(f)
        
        # Crear índice
        index = create_index(df)
    
        # print("Guardando índice y datos...")
        # Guardar índice
        faiss.write_index(index, './bbdd_rag/arxiv_index.faiss')
        
        # print("¡Proceso completado exitosamente!")
        # print(f"Número de artículos procesados: {len(df)}")
        
    except Exception as e:
        print(f"Error durante el proceso: {str(e)}")

if __name__ == "__main__":
    main()
