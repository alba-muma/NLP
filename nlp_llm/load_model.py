import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import torchvision
import json
import argparse
torchvision.disable_beta_transforms_warning()

torch.cuda.empty_cache()

# Ruta a los archivos descargados
model_path = 'C:\\Users\\albam\\.cache\\huggingface\\hub\\models--meta-llama--Llama-3.2-1B\\snapshots\\4e20de362430cd3b72f300e6b0f18e50e7166e08'
# Configurar cuantización de 8 bits
# Configurar el dispositivo (GPU si está disponible)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cuda":
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
else:
    quantization_config = None

# Cargar el tokenizador y el modelo
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map="auto" if device == "cuda" else None  # Esto manejará automáticamente la asignación de memoria
)

# Asignar eos_token como pad_token
tokenizer.pad_token = tokenizer.eos_token

def read_prompt(prompt_file):
    """Lee el contenido de un archivo de prompt"""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    return content

def generate_text(prompt, max_length, max_new_tokens=200):
    """Genera texto basado en un prompt"""
    # Limpiar y formatear el prompt
    prompt = prompt.strip().encode('utf-8', errors='ignore').decode('utf-8')

    # Tokenizar con longitud limitada
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True
    ).to(device)

    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        temperature=0.2,
        num_beams=3,
        do_sample=True,
        # top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # early_stopping=True,
        # length_penalty=-0.5,
        # repetition_penalty=1.5,
        tokenizer=tokenizer,
        stop_strings= ["<STOP>"]
    )
    
    # Decodificar el texto generado
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text.strip()
    
if __name__ == "__main__":
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Generate text based on example prompts')
    parser.add_argument('example_number', type=int, help='Example number to use (e.g., 1 for example_1)')
    args = parser.parse_args()
    
    # Leer el prompt
    prompt_base = read_prompt("./prompts/prompt_0")
    
    # Leer el ejemplo específico
    example_path = f"./prompts/examples/example_{args.example_number}"
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
        "Assistant: "
    )
    # Contar tokens del prompt
    tokens = tokenizer.encode(full_prompt)
    print(len(tokens))

    print("\nInvestigador:")
    print(query)
    generated = generate_text(max_length= len(tokens), prompt=full_prompt)
    print('\nPapers:')
    for i, paper in enumerate(papers_dict["papers"], 1):
        print(f'\t{i}. {paper["title"]}. {paper["abstract"]}')
    start = generated.rfind('User\'s query is in') + len('User\'s query is in ')
    end = generated.rfind(', I shall answer in this language') 
    print('\nIdioma detectado: ', generated[start:end].strip())
    print("\nSistema:")
    # print(generated[len(full_prompt):generated.rfind('<STOP')])
    start = generated.rfind('I shall answer in this language') + len('I shall answer in this language.')
    print(generated[start:generated.rfind('<STOP')].strip())