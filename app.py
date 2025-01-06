import streamlit as st
# Configuración de la página - DEBE SER LA PRIMERA LLAMADA A STREAMLIT
st.set_page_config(
    page_title="Buscador de investigación académica",
    page_icon="📚",
    layout="wide"
)
from search_engine import SearchEngine

# Estilos CSS personalizados
st.markdown("""
    <style>
    .stAlert {
        padding: 0.5rem;
        margin-bottom: 1rem;
    }
    .paper-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        height: 100%;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    </style>
""", unsafe_allow_html=True)

# Inicializar el motor de búsqueda
@st.cache_resource
def get_search_engine():
    return SearchEngine()

# Título principal
st.title("📚 Buscador de investigación académica")
st.markdown("---")

# Crear dos columnas
left_col, right_col = st.columns([1, 1])

with left_col:
    # Barra de búsqueda en la columna izquierda
    st.markdown("### 👁 Investigador:")
    query = st.text_input("", placeholder="Introduce tu consulta...")
    search_button = st.button("🔍 Buscar", type="primary", use_container_width=True)

    # Procesar la búsqueda y mostrar respuesta del sistema
    if search_button and query:
        with st.spinner("Procesando tu consulta..."):
            try:
                engine = get_search_engine()
                results = engine.process_query(query)
                
                # Mostrar información del idioma si existe
                if 'warning' not in results["language_info"]:
                        st.info(f"""
                        🌐 **Idioma detectado:** {results["language_info"]["lang"]}
                        
                        **Consulta traducida:** {results["language_info"]["translated_query"]}
                        """)
                else:
                    st.warning(f"⚠️ {results['language_info']['warning']}")
                
                # Mostrar la respuesta del sistema
                st.markdown("### 💡 Sistema:")
                st.markdown(results["response"])
            except Exception as e:
                st.error(f"Error durante el procesamiento: {str(e)}")
    elif search_button:
        st.warning("⚠️ Por favor, introduce una consulta.")

with right_col:
    # Mostrar los papers en la columna derecha
    if 'results' in locals() and search_button and query and results["papers"]:
        st.markdown("### 📄 Papers Relacionados")
        for paper in results["papers"]:
            with st.container():
                st.markdown(f"""
                <div class="paper-card">
                    <h4>{paper['title']}</h4>
                    <p><small><strong>Similitud</strong>: {paper['similarity']*100:.1f}%</small></p>
                    <p><strong>Abstract</strong>: {paper['abstract']}</p>
                    <p><strong>Resumen</strong>: {paper['summary']}</p>
                    <p><strong>Categorías</strong>: <strong>{paper['categories']}</strong></p>
                </div>
                """, unsafe_allow_html=True)
