import numpy as np
import matplotlib.pyplot as plt
from search import SemanticSearch

def plot_similarity_function(distances, alpha=0.5):
    """Visualiza la función de similitud y las distancias encontradas"""
    # Crear un rango de distancias más fino para la curva
    x = np.linspace(0, max(distances) * 1.2, 1000)
    y = np.exp(-alpha * x)
    
    # Calcular similitudes para las distancias reales
    similarities = np.exp(-alpha * distances)
    
    # Crear la figura
    plt.figure(figsize=(10, 6))
    
    # Plotear la curva continua
    plt.plot(x, y, 'b-', label=f'Función de similitud (α={alpha})')
    
    # Plotear los puntos de las distancias reales
    plt.scatter(distances, similarities, color='red', label='Distancias encontradas')
    
    # Añadir una línea horizontal en y=0.5 (umbral)
    plt.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='Umbral (0.5)')
    
    # Configurar el gráfico
    plt.title('Función de Similitud vs Distancias')
    plt.xlabel('Distancia')
    plt.ylabel('Similitud')
    plt.grid(True, alpha=alpha)
    plt.legend()
    
    # Mostrar el gráfico
    plt.show()

def main():
    # Inicializar el buscador
    searcher = SemanticSearch()
    
    # Hacer una búsqueda de prueba
    query = "neural networks in computer vision"
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
        
        # Mostrar distancias y similitudes
        alpha = 0.25
        print("\nDistancias y similitudes:")
        print("-" * 50)
        for dist in distances[0]:
            similarity = np.exp(-alpha * dist)
            print(f"Distancia: {dist:.3f}, Similitud: {similarity:.3f}")
        
        # Mostrar gráfico
        plot_similarity_function(distances[0], alpha)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
