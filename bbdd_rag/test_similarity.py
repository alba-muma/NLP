import numpy as np
import matplotlib.pyplot as plt
from search import SemanticSearch

def plot_similarity_function(distances):
    """Visualiza la función de similitud y las distancias encontradas"""
    print("\nDistancias y similitudes:")
    print("-" * 50)
    for dist in distances:
        similarity = np.exp(-dist*(dist**2))
        print(f"Distancia: {dist:.3f}, Similitud: {similarity:.3f}")
    
    # Crear un rango de distancias más fino para la curva
    x = np.linspace(0, max(distances) * 1.2, 1000)
    y = np.exp(-x*(x**2))
    
    # Calcular similitudes para las distancias reales
    similarities = np.exp(-distances*(distances**2))
    
    # Crear la figura
    plt.figure(figsize=(10, 6))
    
    # Plotear la curva continua
    plt.plot(x, y, 'b-', label=f'')
    
    # Plotear los puntos de las distancias reales
    plt.scatter(distances, similarities, color='red', label='Distancias encontradas')
    
    # Añadir una línea horizontal con umbral
    umbral = 0.2
    plt.axhline(y=umbral, color='r', linestyle='--', label='Umbral {umbral:.2f}')
        
    # Configurar el gráfico
    plt.title('Función de Similitud vs Distancias')
    plt.xlabel('Distancia')
    plt.ylabel('Similitud')
    plt.grid(True)
    plt.legend()
    
    # Mostrar el gráfico
    plt.show()

def main():
    # Inicializar el buscador
    searcher = SemanticSearch()
    
    # Hacer una búsqueda de prueba
    # query = "neural networks in computer vision"
    query = "Supervised machine learning"
    print(f"\nBuscando: '{query}'")
    print("-" * 50)
    
    # Obtener distancias y mostrar gráfico
    try:
        # Crear embedding de la consulta
        query_embedding = searcher.model.encode(query, convert_to_numpy=True)
        query_embedding = np.array([query_embedding], dtype='float32')
        
        # Buscar los k vecinos más cercanos
        k = searcher.index.ntotal
        distances, indices = searcher.index.search(query_embedding, k)
        
        # Mostrar gráfico
        plot_similarity_function(distances[0])
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
