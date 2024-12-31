import pickle

def view_articles():
    try:
        # Cargar los datos
        print("Cargando datos...")
        with open('arxiv_data.pkl', 'rb') as f:
            df = pickle.load(f)
        
        print(f"\nTotal de artículos: {len(df)}\n")
        print("-" * 100)
        
        # Mostrar cada artículo
        for i, (_, article) in enumerate(df.iterrows(), 1):
            print(f"\nArtículo {i}:")
            print(f"Título: {article['title']}")
            print(f"ID: {article['id']}")
            print(f"Categorías: {article['categories']}")
            print(f"Summary: {article['summary']}")
            print("-" * 100)

    except FileNotFoundError:
        print("Error: No se encontró el archivo arxiv_data.pkl")
        print("Asegúrate de haber ejecutado create_vector_db.py primero")
    except Exception as e:
        print(f"Error al cargar los datos: {str(e)}")

if __name__ == "__main__":
    view_articles()
