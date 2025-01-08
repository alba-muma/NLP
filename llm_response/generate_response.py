import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torchvision
import argparse
torchvision.disable_beta_transforms_warning()
# Redirigir stderr a null
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
stderr = sys.stderr
# sys.stderr = open(os.devnull, 'w')

class LLMManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.model = None
        self.tokenizer = None
        self.model_path = 'C:\\Users\\U01A40E5\\.cache\\huggingface\\hub\\models--meta-llama--Llama-3.2-1B\\snapshots\\4e20de362430cd3b72f300e6b0f18e50e7166e08'
        
        # Configurar cuantización de 8 bits
        self.quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        
        self._initialize()
        self._initialized = True
    
    def _initialize(self):
        """
        Inicializa el modelo y el tokenizer
        """
        # Cargar tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Intentar cargar el modelo en GPU con bitsandbytes
        try:
            torch.cuda.empty_cache()
            print('Cargando Llama-3.2-1B...')
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                quantization_config=self.quantization_config
            )
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception as e:
            # Cargar el modelo sin bitsandbytes y moverlo a la GPU si es posible
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
                self.device = "cuda"
            else:
                self.device = "cpu"

        print(f'Llama-3.2-1B cargado en {self.device}')

    def get_input_tokens(self, prompt):
        """
        Obtiene el número de tokens de entrada para el prompt
        """
        return len(self.tokenizer.encode(prompt))
    
    def generate_text(self, prompt, max_length, max_new_tokens=300):
        """
        Genera texto usando el modelo
        """
        try:
            # Limpiar y formatear el prompt
            prompt = prompt.strip().encode('utf-8', errors='ignore').decode('utf-8')

            # Tokenizar con longitud limitada
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True
            ).to(self.device)

            # Generar respuesta
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.1,
                    # top_p=0.9,
                    num_beams=3,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    tokenizer=self.tokenizer
                )

            # Decodificar y retornar la respuesta
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split('Response:')[4].split('<STOP>')[0]
            return response
            
        except Exception as e:
            print(f"Error en la generación: {str(e)}")
            return ""

# Crear una instancia global del LLMManager
global_llm = LLMManager()

def read_prompt(prompt_path):
    """
    Lee un prompt desde un archivo
    """
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

# Funciones helper que usan la instancia global
def generate_text(prompt, max_length, max_new_tokens=300):
    return global_llm.generate_text(prompt, max_length, max_new_tokens)

def get_input_tokens(prompt):
    return global_llm.get_input_tokens(prompt)

if __name__ == "__main__":
    # Vaciar la memoria de la GPU
    torch.cuda.empty_cache()

    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Generate text based on example prompts')
    parser.add_argument('example_number', type=int, help='Example number to use (e.g., 1 for example_1)')
    args = parser.parse_args()

    # Leer el prompt
    prompt = 4
    prompt_base = read_prompt(f"./llm_response/prompts/prompt_{prompt}")
    
    # Leer el ejemplo específico
    example_path = f"./llm_response/prompts/examples/example_{args.example_number}"
    try:
        with open(example_path, 'r', encoding='utf-8') as f:
            example_content = f.read()
            # Evaluar el contenido del archivo para obtener query y papers_dict
            example_vars = {}
            exec(example_content, {}, example_vars)
            query = example_vars.get('query')
            papers_dict = example_vars.get('papers_dict')
            
            if not query or not papers_dict:
                raise ValueError("El archivo de ejemplo debe contener 'query' y 'papers_dict'")
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de ejemplo {example_path}")
        exit(1)
    except Exception as e:
        print(f"Error al leer el archivo de ejemplo: {str(e)}")
        exit(1)
    
    # Generar el prompt completo
    user_query = f"{papers_dict}\nUser: {query}"
    full_prompt = (
        prompt_base + '\n' +
        "Papers: " + 
        user_query + '\n' +
        "Response: "
    )

    print("\nInvestigador:")
    print(query)
    generated = generate_text(max_length= get_input_tokens(full_prompt), prompt=full_prompt)
    print('\nPapers:')
    for i, paper in enumerate(papers_dict["papers"], 1):
        print(f'\t{i}. {paper["title"]}. {paper["summary"]}')
    print("\nSistema:")
    print(generated)

    # Restaurar stderr al final
    sys.stderr = stderr