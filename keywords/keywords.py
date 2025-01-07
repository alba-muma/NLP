import spacy
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def preprocess_text_spacy(text, nlp):

    if not text:  
        return ""
    try:
        doc = nlp(text)
        tokens = [token.text for token in doc if token.is_alpha and token.pos_ in ("NOUN", "PROPN", "ADJ")]
        return " ".join(tokens)
    except Exception as e:
        print(f"Error al procesar el texto: {text[:50]}... - {e}")
        return ""

def extract_keywords_per_document(row_idx, tfidf_matrix, feature_names, max_keywords=10):

    row_vector = tfidf_matrix[row_idx].toarray().flatten()
    sorted_indices = row_vector.argsort()[::-1]
    keywords = [feature_names[idx] for idx in sorted_indices[:max_keywords]]
    return keywords

def generate_keywords(df, batch_size=1000):

    print("Preprocesando texto con spaCy para palabras clave...")
    nlp = spacy.load("en_core_web_md")
    tqdm.pandas()

    combined_texts = (df['title'].fillna("") + " " + df['abstract'].fillna("")).tolist()

    processed_texts = []
    for start in tqdm(range(0, len(combined_texts), batch_size), desc="Procesando lotes de textos", unit="lote"):
        end = start + batch_size
        batch = [preprocess_text_spacy(text, nlp) for text in combined_texts[start:end]]
        processed_texts.extend(batch)

    print("Calculando matriz TF-IDF y palabras clave...")
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(processed_texts)
    tfidf_keywords = tfidf.get_feature_names_out()

    df['keywords'] = [
        extract_keywords_per_document(i, tfidf_matrix, tfidf_keywords, max_keywords=10)
        for i in tqdm(range(tfidf_matrix.shape[0]), desc="Extrayendo palabras clave", unit="doc")
    ]

    return df
