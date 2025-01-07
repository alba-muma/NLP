from gensim.models.ldamodel import LdaModel
import ast
import pandas as pd
from tqdm import tqdm
from gensim.corpora.dictionary import Dictionary

def extract_topics_with_threshold(row, threshold, topic_dict):
    """
    Extrae los tópicos principales de un documento con un umbral de probabilidad.
    
    :param row: Distribución de tópicos del documento.
    :param threshold: Umbral de probabilidad para considerar un tópico.
    :param topic_dict: Diccionario que mapea tópicos a palabras clave.
    :return: Lista de tópicos principales separados por comas.
    """
    top_topics = sorted(row, key=lambda x: x[1], reverse=True)[:3]
    selected_topics = [topic_dict[int(topic)] for topic, prob in top_topics if prob > threshold]
    return ', '.join(selected_topics)

def perform_topic_modeling(df, num_topics=80, passes=10, threshold=0.01):
    """
    Realiza modelado de tópicos usando LDA (Latent Dirichlet Allocation)
    
    :param df: DataFrame con los datos a procesar.
    :param num_topics: Número de tópicos a identificar.
    :param passes: Número de iteraciones para entrenar el modelo.
    :param threshold: Umbral de probabilidad para considerar un tópico.
    :return: DataFrame con la información de tópicos agregada.
    """
    print("Iniciando el modelado de tópicos...")

    # Cargar textos procesados
    with open('./lda/processed_texts.txt', 'r', encoding='utf-8') as f:
        processed_texts = [line.strip() for line in f if line.strip()]

    # Crear diccionario y corpus
    processed_texts_split = [text.split() for text in processed_texts]
    dictionary = Dictionary(processed_texts_split)
    corpus_bow = [dictionary.doc2bow(text) for text in processed_texts_split]

    print("Entrenando el modelo LDA...")
    ldamodel = LdaModel(corpus=corpus_bow, id2word=dictionary, num_topics=num_topics, passes=passes, random_state=42)

    print("Generando la distribución de tópicos para cada documento...")
    all_doc_topics = [
        [(f"{topic_id}", prob) for topic_id, prob in ldamodel.get_document_topics(bow, minimum_probability=0.001)]
        for bow in corpus_bow
    ]

    df = df.copy()
    df['topic_distribution'] = all_doc_topics

    print("Mapeando tópicos a palabras clave...")
    keywords_topics = pd.read_csv('./lda/cleaned_keywords_topics.csv')
    topic_dict = dict(zip(
        keywords_topics['Tópico'].str.extract(r'(\d+)')[0].astype(int),
        keywords_topics['Keywords']
    ))

    df['main_topics'] = df['topic_distribution'].apply(
        lambda x: extract_topics_with_threshold(x, threshold, topic_dict)
    )

    print("Modelado de tópicos completado.")
    return df
